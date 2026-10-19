"""Explicit synthetic plant and observations. Never a fallback for live IO."""

import copy
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from robot_control.closed_loop_observation import (
    CAMERAS,
    JOINT_NAMES,
    SOURCES,
    ObservationPending,
    validated_observation,
)
from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import ExecutionFeedback


class SyntheticGeometry:
    """Small linear test geometry. All reports label it synthetic."""

    def __init__(self):
        self.bounds = np.tile([-2.0, 2.0], (14, 1))
        self.parameters = dict(collision_minimum_distance=0.005)
        self.manifest = dict(
            synthetic=True, same_installed_qp_core=False, solver="linear_test_geometry"
        )

    def poses(self, q):
        p = np.array([q[:6], q[7:13]], dtype=float)
        p[:, 0] += 0.4
        p[:, 2] += 0.8
        p[0, 1] += 0.2
        p[1, 1] -= 0.2
        return p

    def collisions(self, q):
        return [dict(pair=["synthetic_left", "synthetic_right"], distance_m=0.1)]

    def rollout(self, actions, seed, observed_state, output, *, prefix_steps=50):
        n = int(np.ceil(prefix_steps / 30 / 0.002 - 1e-9))
        ids = np.floor(np.arange(n) * 0.002 * 30 + 1e-8).astype(int)
        q = np.tile(seed, (n, 1))
        current = self.poses(seed)
        q[:, :6] += actions[ids, :6] - current[0]
        q[:, 7:13] += actions[ids, 7:13] - current[1]
        np.savez_compressed(
            Path(output) / "qp_trajectory.npz",
            joint_positions=q,
            target_indices=ids,
            seed=seed,
            dt=0.002,
        )
        return dict(
            status="passed_scoped_qp",
            reasons=[],
            simulated_ticks=n,
            manifest=self.manifest,
        )


class SimulatedIO:
    production = False
    bounded_cached_feedback = True

    def __init__(self):
        self.lock = threading.RLock()
        self.q = np.zeros(14)
        self.target = self.q.copy()
        self.v = np.zeros(14)
        self.input = 0
        self.mode = 1
        self.boot = uuid.uuid4().hex
        self.session = None
        self.grip = np.array([0.5, 0.5])
        self.grip_state = "idle"
        self.generation = 0
        self.last_request_id = None
        self.last_seq = -1
        self.fail_side = None
        self.stale = False
        self.unexpected = False
        self.gripper_delay = 0.0
        self.error = None
        self.messages = []
        self.claims = self.releases = 0
        self.authorized = None
        self.seed = None
        self.closed = False

    def gripper_status(self):
        with self.lock:
            return dict(
                boot_id=self.boot,
                session_id=self.session,
                reserved=self.session is not None,
                healthy=True,
                state=self.grip_state,
                applied_target=self.grip.tolist(),
                actual=self.grip.tolist(),
                motor_feedback_age=[0.0, 0.0],
                motor_velocities=[0.0, 0.0],
                calibration=[[0.0, 1.6], [0.0, 1.6]],
                target_generation=self.generation,
                last_request_id=self.last_request_id,
                result=dict(ok=True),
            )

    def feedback(self):
        with self.lock:
            now = time.monotonic()
            sample = now - 10 if self.stale else now
            return ExecutionFeedback(
                sample,
                now - sample,
                self.q.copy(),
                self.v.copy(),
                self.target.copy(),
                sample,
                self.input,
                self.error is None,
                self.mode,
                self.gripper_status(),
                sample,
                self.unexpected,
            )

    def bind_stationary_target(self, q, grip):
        if self.seed is not None:
            raise ControlFault("session_already_bound")
        self.seed = np.array(q, copy=True)

    def arm_segment(self, segment):
        self.authorized = segment.digest()

    def validate_targets(self, q, grip):
        if (
            self.seed is None
            or np.asarray(q).shape != (14,)
            or not np.isfinite(q).all()
        ):
            raise ControlFault("invalid_or_unbound_target")

    def select_input(self, value):
        self.input = value

    def publish_arm(self, side, q):
        with self.lock:
            if self.fail_side == side:
                raise ControlFault("injected_arm_failure:" + side)
            i = 0 if side == "left" else 7
            self.target[i : i + 7] = q
            self.q[i : i + 7] = q
            self.messages.append(dict(time=time.monotonic(), side=side, q=list(q)))

    def gripper_command(self, request):
        if self.gripper_delay:
            time.sleep(self.gripper_delay)
        with self.lock:
            action = request["action"]
            if request["boot_id"] != self.boot:
                raise ControlFault("wrong_boot")
            if action == "claim":
                if self.session is not None:
                    raise ControlFault("already_owned")
                self.claims += 1
                self.session = str(uuid.UUID(request["session_id"]))
                self.grip_state = "active"
            else:
                if request["session_id"] != self.session:
                    raise ControlFault("wrong_session")
                if request["seq"] <= self.last_seq:
                    raise ControlFault("old_gripper_sequence")
                if action == "target":
                    if (
                        self.grip_state != "active"
                        or request["target_generation"] != self.generation
                    ):
                        raise ControlFault("invalid_target_generation")
                    self.grip[0 if request["side"] == "left" else 1] = request["value"]
                elif action == "hold":
                    self.grip_state = "held"
                    self.generation += 1
                elif action == "release":
                    if self.grip_state != "held":
                        raise ControlFault("release_without_hold")
                    self.releases += 1
                    self.session = None
                    self.grip_state = "idle"
                else:
                    raise ControlFault("unknown_gripper_action")
            self.last_seq = request["seq"]
            self.last_request_id = request["request_id"]
            return self.gripper_status()

    def heartbeat(self, request):
        if request["boot_id"] != self.boot or request["session_id"] != self.session:
            raise ControlFault("invalid_heartbeat")

    def close(self):
        self.closed = True


class SimulatedObservationSource:
    def __init__(self, io, config, freeze_after=None):
        self.io, self.config = io, config
        self.freeze_after = freeze_after
        self.serial = 0
        self.geometry = SyntheticGeometry()
        self.frozen = False
        self.previous = None

    def read_after(self, barrier, previous_id):
        if (
            self.frozen
            or self.freeze_after is not None
            and self.serial >= self.freeze_after
        ) and self.previous:
            meta, state, images = self.previous
        else:
            f = self.io.feedback()
            if (
                f.received_at <= barrier.monotonic
                or f.gripper_received_at <= barrier.monotonic
            ):
                raise ObservationPending("simulated_plant_feedback_before_hold")
            self.serial += 1
            now, wall = time.monotonic(), time.time()
            pose = self.geometry.poses(f.joints)
            state = np.r_[
                pose[0],
                f.gripper["actual"][0],
                np.zeros(6),
                pose[1],
                f.gripper["actual"][1],
                np.zeros(6),
            ].astype(np.float32)
            sources = {k: dict(source_time=wall, received_mono=now) for k in SOURCES}
            meta = dict(
                snapshot_id=f"synthetic-{self.serial}",
                synthetic=True,
                errors=[],
                sources=sources,
                joints=dict(
                    names=JOINT_NAMES,
                    positions=f.joints.tolist(),
                    source=dict(source_time=wall, received_mono=now),
                ),
                control_feedback=dict(
                    joints=dict(
                        positions=f.joints.tolist(),
                        velocities=f.velocities.tolist(),
                        source_time=wall,
                        received_mono=now,
                    ),
                    left_gripper=dict(
                        position_rad=float(f.gripper["actual"][0]),
                        velocity_rad_s=0.0,
                        received_mono=now,
                    ),
                    right_gripper=dict(
                        position_rad=float(f.gripper["actual"][1]),
                        velocity_rad_s=0.0,
                        received_mono=now,
                    ),
                ),
                gripper_target_source=dict(source_time=wall, received_mono=now),
                gripper_applied_target=f.gripper["applied_target"],
                motor_feedback_age=[0.0, 0.0],
            )
            images = {
                key: np.full(
                    (480, 640, 3), int((float(f.joints[0]) * 1e5 + 50) % 255), np.uint8
                )
                for key in CAMERAS
            }
            self.previous = (
                copy.deepcopy(meta),
                state.copy(),
                {k: v.copy() for k, v in images.items()},
            )
        return validated_observation(
            meta, state, images, barrier, previous_id, self.config
        )


class SyntheticPolicy:
    def __init__(self, delay=0.05, move_gripper=True, fault=None):
        self.delay, self.move_gripper, self.fault = delay, move_gripper, fault
        self.requests = []

    def predict(self, raw, request_id):
        self.requests.append(
            dict(id=request_id, state=np.array(raw["state"], copy=True))
        )
        time.sleep(self.delay)
        actions = np.tile(
            np.r_[raw["state"][:7], raw["state"][13:20]], (1, 50, 1)
        ).astype(float)
        actions[0, :, 0] += np.linspace(0.00008, 0.0002, 50)
        actions[0, :, 7] += np.linspace(0.00005, 0.0001, 50)
        if self.move_gripper:
            actions[0, :, 6] += 0.0001
        if not request_id.startswith("warmup-"):
            if self.fault == "wrong_id":
                request_id = "wrong-response"
            elif self.fault == "nonfinite":
                actions[0, 0, 0] = np.nan
            elif self.fault == "rpc_error":
                raise RuntimeError("injected_RPC_error")
        return dict(actions=actions, request_id=request_id, synthetic=True)
