"""V1 contracts. Importing this module performs no IO or robot operations."""

import hashlib
import json
import math
import threading
import time
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Protocol

import numpy as np

from robot_control.command_bridge import ControlFault


@dataclass(frozen=True)
class LoopConfig:
    schema_version: int = 1
    prediction_horizon: int = 50
    action_period_s: float = 1 / 30
    control_period_s: float = 0.002
    execution_horizon: int = 3
    max_cycles: int = 3
    max_task_duration_s: float = 30.0
    observation_timeout_s: float = 2.0
    rpc_timeout_s: float = 2.0
    warmup_timeout_s: float = 10.0
    prepare_timeout_s: float = 1.0
    max_prediction_age_s: float = 2.5
    max_observation_age_s: float = 0.5
    max_source_skew_s: float = 0.15
    clock_tolerance_s: float = 0.05
    feedback_age_s: float = 0.2
    settle_time_s: float = 0.2
    source_ramp_s: float = 3.2
    acquire_timeout_s: float = 6.0
    hold_timeout_s: float = 1.5
    heartbeat_period_s: float = 0.05
    supervision_timeout_s: float = 0.25
    browser_disconnect_timeout_s: float = 5.0
    max_lateness_s: float = 0.0015
    max_joint_displacement_rad: float = 0.01
    max_joint_speed_rad_s: float = 0.1
    max_joint_acceleration_rad_s2: float = 1.0
    max_tcp_displacement_m: float = 0.005
    max_tcp_rotation_rad: float = 0.03
    max_tcp_tracking_m: float = 0.002
    max_tcp_tracking_rad: float = 0.03
    seed_error_rad: float = 0.002
    tracking_error_rad: float = 0.005
    stop_speed_rad_s: float = 0.02
    gripper_stop_speed_rad_s: float = 0.03
    gripper_tracking_error_rad: float = 0.05
    max_gripper_displacement: float = 0.02
    max_gripper_speed_per_s: float = 0.03
    max_gripper_correction: float = 0.05
    gripper_receipt_timeout_s: float = 0.2
    gripper_policy: str = "model"
    max_log_mb: int = 256

    def __post_init__(self):
        ints = dict(
            schema_version=(1, 1),
            prediction_horizon=(50, 50),
            execution_horizon=(1, 3),
            max_cycles=(1, 100),
            max_log_mb=(1, 4096),
        )
        for name, value in asdict(self).items():
            if name in ints:
                lo, hi = ints[name]
                if type(value) is not int or not lo <= value <= hi:
                    raise ValueError("Invalid integer configuration: " + name)
            elif name == "gripper_policy":
                if value not in ("model", "fixed_diagnostic"):
                    raise ValueError("Invalid gripper_policy")
            elif (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid positive finite configuration: " + name)
        if abs(self.action_period_s - 1 / 30) > 1e-12 or self.control_period_s != 0.002:
            raise ValueError("V1 requires 30Hz targets and 2ms joint ticks")
        if self.max_lateness_s >= self.control_period_s:
            raise ValueError("lateness must be shorter than one control tick")
        if self.heartbeat_period_s * 2 >= self.supervision_timeout_s:
            raise ValueError("supervision lease needs heartbeat margin")
        if self.source_ramp_s + self.settle_time_s >= self.acquire_timeout_s:
            raise ValueError("acquire timeout must cover source ramp and settling")
        if self.settle_time_s >= self.hold_timeout_s:
            raise ValueError("hold timeout must allow settling")
        if self.execution_horizon * self.action_period_s >= self.max_prediction_age_s:
            raise ValueError("prediction age cannot cover execution")
        if self.gripper_stop_speed_rad_s > 0.03:
            raise ValueError("V1 does not inherit the measured-hold monitor override")

    @classmethod
    def load(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_dict(cls, data):
        extra = set(data) - {f.name for f in fields(cls)}
        if extra:
            raise ValueError("Unknown configuration fields: " + ",".join(sorted(extra)))
        return cls(**data)

    def digest(self):
        return hashlib.sha256(
            json.dumps(asdict(self), sort_keys=True).encode()
        ).hexdigest()


@dataclass(frozen=True)
class Barrier:
    monotonic: float
    unix: float

    @classmethod
    def now(cls):
        mono = time.monotonic()
        return cls(mono, time.time())


@dataclass(frozen=True)
class CycleIdentity:
    run_id: str
    cycle: int
    generation: int
    observation_id: str
    request_id: str


@dataclass(frozen=True)
class Observation:
    id: str
    observed_at: float
    source_times: dict
    state: np.ndarray
    joints: np.ndarray
    velocities: np.ndarray
    gripper_targets: np.ndarray
    images: dict
    metadata: dict

    def request(self, prompt, seed):
        return {**self.images, "state": self.state, "prompt": prompt, "seed": seed}


@dataclass(frozen=True)
class Segment:
    identity: CycleIdentity
    observed_at: float
    dt: float
    seed: np.ndarray
    seed_velocity: np.ndarray
    initial_grippers: np.ndarray
    joints: np.ndarray
    grippers: np.ndarray
    target_indices: np.ndarray
    brake_paths: tuple
    checks: dict

    @property
    def duration_s(self):
        return len(self.joints) * self.dt

    def digest(self):
        h = hashlib.sha256(
            json.dumps(
                dict(
                    identity=asdict(self.identity),
                    observed_at=self.observed_at,
                    dt=self.dt,
                ),
                sort_keys=True,
            ).encode()
        )
        for array in (
            self.seed,
            self.seed_velocity,
            self.initial_grippers,
            self.joints,
            self.grippers,
            self.target_indices,
            *self.brake_paths,
        ):
            a = np.ascontiguousarray(array)
            h.update(str((a.shape, a.dtype.str)).encode())
            h.update(a.tobytes())
        return h.hexdigest()


class Cancellation:
    def __init__(self):
        self.event = threading.Event()
        self.lock = threading.Lock()
        self.reason = None
        self.generation = 0

    def cancel(self, reason="user_cancelled"):
        with self.lock:
            if not self.event.is_set():
                self.reason = reason
                self.generation += 1
                self.event.set()

    def check(self, generation=None):
        if (
            self.event.is_set()
            or generation is not None
            and generation != self.generation
        ):
            raise ControlFault(self.reason or "generation_invalidated")


class ObservationSource(Protocol):
    def read_after(self, barrier: Barrier, previous_id: str | None) -> Observation: ...


class SegmentPreparer(Protocol):
    def prepare(
        self,
        observation: Observation,
        prediction: dict,
        identity: CycleIdentity,
        folder: Path,
    ) -> Segment: ...


class ExecutionSession(Protocol):
    def acquire(self, owner: str) -> Barrier: ...
    def execute(self, segment: Segment, cancellation: Cancellation) -> dict: ...
    def wait_hold(self, cancellation: Cancellation) -> Barrier: ...
    def supervise(self) -> None: ...
    def stop(self, reason: str) -> dict: ...
    def release(self) -> None: ...
    def status(self) -> dict: ...
