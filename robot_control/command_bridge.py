"""Bounded action execution against an explicit controller contract.

No ROS/network imports. The caller supplies monotonic time and a backend with
verified acquisition, fresh feedback, submission and per-channel stop receipts.
"""

from dataclasses import dataclass
import math
import numpy as np
from scipy.spatial.transform import Rotation
from robot_control.action_adapter import prepare_actions


class ControlFault(RuntimeError):
    pass


@dataclass(frozen=True)
class Feedback:
    state: np.ndarray
    received_at: float
    owner_token: str
    gripper_owner_token: str
    healthy: bool = True
    ik_healthy: bool = True


@dataclass(frozen=True)
class Prediction:
    request_id: str
    observed_at: float
    actions: np.ndarray
    checks_passed: bool


def pose_error(a, b):
    return (
        float(np.linalg.norm(a[:3] - b[:3])),
        float(
            (
                Rotation.from_euler("xyz", b[3:]).inv()
                * Rotation.from_euler("xyz", a[3:])
            ).magnitude()
        ),
    )


def current_action(state):
    state = np.asarray(state, dtype=np.float64)
    if state.shape != (26,) or not np.isfinite(state).all():
        raise ControlFault("invalid_feedback")
    return np.r_[state[:7], state[13:20]]


def valid_limits(limits):
    for key in [
        "control_hz",
        "feedback_max_age_s",
        "prediction_max_age_s",
        "rpc_timeout_s",
        "max_lateness_s",
        "tracking_position_m",
        "tracking_rotation_rad",
        "tracking_gripper",
        "max_position_step_m",
        "max_rotation_step_rad",
        "max_linear_speed_m_s",
        "max_angular_speed_rad_s",
        "max_linear_acceleration_m_s2",
        "max_angular_acceleration_rad_s2",
        "max_gripper_speed_s",
        "max_hold_s",
        "max_gripper_correction",
    ]:
        value = limits.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid " + key)
    if (
        type(limits.get("execution_horizon")) is not int
        or not 1 <= limits["execution_horizon"] <= 50
    ):
        raise ValueError("Invalid horizon")
    for side in ["left", "right"]:
        bounds = np.asarray(limits["workspace_m"][side])
        if (
            bounds.shape != (2, 3)
            or not np.isfinite(bounds).all()
            or np.any(bounds[0] >= bounds[1])
        ):
            raise ValueError("Invalid workspace")


class CommandBridge:
    def __init__(self, backend, limits):
        valid_limits(limits)
        self.backend, self.limits = backend, limits
        self.state = "DISCONNECTED"
        self.token = None
        self.events = []
        self.pending = None
        self.actions = None
        self.seen = set()
        self.next_tick = None
        self.target = None
        self.stop_receipts = {}
        self.sent = 0
        self.fault_reason = None
        self.held_since = None

    def event(self, now, event, **fields):
        self.events.append({"time": now, "event": event, "state": self.state, **fields})

    def feedback(self, now):
        fb = self.backend.read(now)
        if (
            not math.isfinite(fb.received_at)
            or not 0 <= now - fb.received_at <= self.limits["feedback_max_age_s"]
        ):
            raise ControlFault("stale_feedback")
        if not fb.healthy:
            raise ControlFault("controller_unhealthy")
        if not fb.ik_healthy:
            raise ControlFault("ik_failure")
        if fb.owner_token != self.token or fb.gripper_owner_token != self.token:
            raise ControlFault("ownership_lost")
        current_action(fb.state)
        return fb

    def connect(self, now):
        if self.state != "DISCONNECTED":
            raise ControlFault("session_not_resettable")
        try:
            self.token = self.backend.acquire(now)
            if not isinstance(self.token, str) or not self.token:
                raise ControlFault("ownership_not_confirmed")
            fb = self.feedback(now)
            self.target = current_action(fb.state)
            self.validate_sequence(self.target[None], self.target)
            self.state = "HOLDING"
            self.held_since = now
            self.next_tick = now
            self.event(now, "acquired")
        except Exception as exc:
            self.fail(now, str(exc))
            raise

    def validate_sequence(self, actions, start):
        a = np.asarray(actions, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 14 or not np.isfinite(a).all():
            raise ControlFault("invalid_actions")
        if np.any(a[:, [6, 13]] < 0) or np.any(a[:, [6, 13]] > 1):
            raise ControlFault("gripper_out_of_range")
        extended = np.vstack((start, a))
        period = 1 / 30
        for side, offset in [("left", 0), ("right", 7)]:
            bounds = np.asarray(self.limits["workspace_m"][side])
            positions = a[:, offset : offset + 3]
            if np.any(positions < bounds[0]) or np.any(positions > bounds[1]):
                raise ControlFault("workspace_" + side)
            poses = extended[:, offset : offset + 6]
            linear = np.diff(poses[:, :3], axis=0) / period
            rotations = Rotation.from_euler("xyz", poses[:, 3:])
            # World-frame angular increments avoid Euler wrap discontinuities.
            angular = (rotations[1:] * rotations[:-1].inv()).as_rotvec() / period
            if np.any(
                np.linalg.norm(linear, axis=1) * period
                > self.limits["max_position_step_m"]
            ):
                raise ControlFault("position_step")
            if np.any(
                np.linalg.norm(angular, axis=1) * period
                > self.limits["max_rotation_step_rad"]
            ):
                raise ControlFault("rotation_step")
            if np.any(
                np.linalg.norm(linear, axis=1) > self.limits["max_linear_speed_m_s"]
            ):
                raise ControlFault("linear_speed")
            if np.any(
                np.linalg.norm(angular, axis=1) > self.limits["max_angular_speed_rad_s"]
            ):
                raise ControlFault("angular_speed")
            for values, limit, name in [
                (
                    linear,
                    self.limits["max_linear_acceleration_m_s2"],
                    "linear_acceleration",
                ),
                (
                    angular,
                    self.limits["max_angular_acceleration_rad_s2"],
                    "angular_acceleration",
                ),
            ]:
                acceleration = (
                    np.diff(
                        np.vstack((np.zeros((1, 3)), values, np.zeros((1, 3)))), axis=0
                    )
                    / period
                )
                if np.any(np.linalg.norm(acceleration, axis=1) > limit):
                    raise ControlFault(name)
        if np.any(
            np.abs(np.diff(extended[:, [6, 13]], axis=0)) / period
            > self.limits["max_gripper_speed_s"]
        ):
            raise ControlFault("gripper_speed")

    def request(self, request_id, observed_at, now):
        if self.state != "HOLDING":
            raise ControlFault("not_holding")
        if not request_id or request_id in self.seen:
            raise ControlFault("reused_request")
        if (
            not math.isfinite(observed_at)
            or not 0 <= now - observed_at <= self.limits["feedback_max_age_s"]
        ):
            raise ControlFault("stale_request")
        self.seen.add(request_id)
        self.pending = (request_id, observed_at, now)
        self.state = "WAITING"
        self.event(now, "prediction_requested", request_id=request_id)

    def submit(self, prediction, now):
        if self.state != "WAITING" or self.pending is None:
            raise ControlFault("no_pending_request")
        try:
            request_id, observed_at, started = self.pending
            if (
                prediction.request_id != request_id
                or prediction.observed_at != observed_at
            ):
                raise ControlFault("prediction_identity_mismatch")
            if now - started > self.limits["rpc_timeout_s"]:
                raise ControlFault("rpc_timeout")
            if not 0 <= now - observed_at <= self.limits["prediction_max_age_s"]:
                raise ControlFault("prediction_expired")
            if prediction.checks_passed is not True:
                raise ControlFault("prediction_checks_failed")
            if np.asarray(prediction.actions).shape != (50, 14):
                raise ControlFault("invalid_horizon")
            fb = self.feedback(now)
            start = current_action(fb.state)
            # Check all 50 targets before taking a prefix; do not hide invalid tails.
            try:
                candidate, adaptation = prepare_actions(
                    np.asarray(prediction.actions)[None],
                    self.limits["max_gripper_correction"],
                )
            except ValueError as exc:
                raise ControlFault("invalid_actions") from exc
            self.event(now, "action_preparation", details=adaptation)
            if candidate is None:
                raise ControlFault("gripper_correction_excessive")
            self.validate_sequence(candidate[0], start)
            k = self.limits["execution_horizon"]
            self.actions = np.array(candidate[0, :k], dtype=np.float64, copy=True)
            # The prefix ends in a hold; validate its stop transition separately.
            self.validate_sequence(self.actions, start)
            self.observed_at = observed_at
            self.execution_started = now
            self.pending = None
            self.state = "EXECUTING"
            self.event(now, "prediction_accepted", request_id=request_id, steps=k)
        except Exception as exc:
            self.fail(now, str(exc))
            raise

    def tick(self, now):
        if self.state not in ["HOLDING", "WAITING", "EXECUTING"]:
            return
        try:
            if not math.isfinite(now) or now < self.next_tick - 1e-9:
                raise ControlFault("clock_regression_or_early_tick")
            if now - self.next_tick > self.limits["max_lateness_s"]:
                raise ControlFault("control_deadline_missed")
            self.next_tick += 1 / self.limits["control_hz"]
            fb = self.feedback(now)
            actual = current_action(fb.state)
            for offset in [0, 7]:
                p, r = pose_error(
                    actual[offset : offset + 6], self.target[offset : offset + 6]
                )
                if (
                    p > self.limits["tracking_position_m"]
                    or r > self.limits["tracking_rotation_rad"]
                ):
                    raise ControlFault("tracking_error")
                if (
                    abs(actual[offset + 6] - self.target[offset + 6])
                    > self.limits["tracking_gripper"]
                ):
                    raise ControlFault("gripper_tracking_error")
            if (
                self.state == "WAITING"
                and now - self.pending[2] > self.limits["rpc_timeout_s"]
            ):
                raise ControlFault("rpc_timeout")
            if (
                self.state in ["HOLDING", "WAITING"]
                and now - self.held_since > self.limits["max_hold_s"]
            ):
                raise ControlFault("hold_timeout")
            if self.state == "EXECUTING":
                if now - self.observed_at > self.limits["prediction_max_age_s"]:
                    raise ControlFault("prediction_expired")
                index = int(math.floor((now - self.execution_started) * 30 + 1e-8))
                if index >= len(self.actions):
                    self.actions = None
                    self.state = "HOLDING"
                    self.held_since = now
                    self.event(now, "chunk_complete")
                else:
                    self.target = self.actions[index].copy()
            # Independent heartbeat/hold ticks do not refresh observed_at.
            receipt = self.backend.write(self.target.copy(), self.token, now)
            if receipt != {"left": True, "right": True, "grippers": True}:
                raise ControlFault("partial_or_rejected_submission")
            self.sent += 1
            self.event(now, "submitted", target=self.target.tolist(), mode=self.state)
        except Exception as exc:
            self.fail(now, str(exc))

    def fail(self, now, reason):
        if self.state in ["STOPPED", "FAULT_UNCONFIRMED"]:
            return
        self.fault_reason = reason
        self.pending = None
        self.actions = None
        self.state = "STOPPING"
        if self.token is None:
            self.state = "STOPPED"
            self.event(now, "not_acquired_no_stop_sent", reason=reason)
            return
        for channel in ["left", "right", "grippers"]:
            try:
                self.stop_receipts[channel] = (
                    self.backend.stop_channel(channel, self.token, now) is True
                )
            except Exception:
                self.stop_receipts[channel] = False
        stopped = all(self.stop_receipts.values())
        self.state = "STOPPED" if stopped else "FAULT_UNCONFIRMED"
        self.event(now, "stop", reason=reason, receipts=dict(self.stop_receipts))
        if stopped and self.token:
            try:
                self.backend.release(self.token, now)
            except Exception as exc:
                self.state = "FAULT_UNCONFIRMED"
                self.event(now, "release_failed", error=str(exc))

    def close(self, now):
        if self.state not in ["STOPPED", "FAULT_UNCONFIRMED", "DISCONNECTED"]:
            self.fail(now, "session_closed")
        self.backend.close()
