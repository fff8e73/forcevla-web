"""In-process exclusive session gate for a future mux input-3 adapter.

This gate owns only cooperating clients. It cannot stop external ROS publishers
or prove robot stop. Every trip requires an independently verified stop path.
"""

from dataclasses import dataclass
import math
import secrets
import threading

import numpy as np


class GateRejected(RuntimeError):
    pass


@dataclass(frozen=True)
class JointEnvelope:
    token: str
    sequence: int
    command_time: float
    observation_time: float
    joint_positions: np.ndarray


class JointSourceGuard:
    def __init__(
        self,
        bounds,
        velocity_limits,
        *,
        lease_s=0.25,
        command_max_age_s=0.05,
        prediction_max_age_s=2.5,
        feedback_max_age_s=0.1,
        max_joint_step_rad=0.05,
    ):
        self.bounds = np.asarray(bounds, dtype=float).copy()
        self.velocity = np.asarray(velocity_limits, dtype=float).copy()
        if (
            self.bounds.shape != (14, 2)
            or self.velocity.shape != (14,)
            or not np.isfinite(self.bounds).all()
            or not np.isfinite(self.velocity).all()
        ):
            raise ValueError("Invalid robot limits")
        if np.any(self.bounds[:, 0] >= self.bounds[:, 1]) or np.any(self.velocity <= 0):
            raise ValueError("Invalid robot bounds")
        for value in [
            lease_s,
            command_max_age_s,
            prediction_max_age_s,
            feedback_max_age_s,
            max_joint_step_rad,
        ]:
            if (
                type(value) not in (int, float)
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError("Invalid budgets")
        self.lease_s = lease_s
        self.command_max_age_s = command_max_age_s
        self.prediction_max_age_s = prediction_max_age_s
        self.feedback_max_age_s = feedback_max_age_s
        self.max_step = max_joint_step_rad
        self.lock = threading.Lock()
        self.state = "IDLE"
        self.token = None
        self.owner = None
        self.sequence = -1
        self.deadline = None
        self.last_time = None
        self.last_q = None
        self.last_command_time = None
        self.reason = None

    def validate_q(self, value):
        q = np.array(value, dtype=np.float64, copy=True)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise GateRejected("invalid_joint_positions")
        if np.any(q < self.bounds[:, 0]) or np.any(q > self.bounds[:, 1]):
            raise GateRejected("joint_limit")
        return q

    def trip(self, reason):
        self.state = "TRIPPED"
        self.reason = reason
        raise GateRejected(reason)

    def clock(self, now):
        if (
            not math.isfinite(now)
            or self.last_time is not None
            and now < self.last_time
        ):
            self.trip("monotonic_clock_regression")
        self.last_time = now

    def claim(self, owner, now, measured_q, feedback_time, selected_input):
        with self.lock:
            if self.state != "IDLE":
                raise GateRejected("session_busy_or_tripped")
            if not owner:
                raise GateRejected("missing_owner")
            if selected_input != 3:
                raise GateRejected("input_3_not_selected")
            if (
                not math.isfinite(now)
                or not math.isfinite(feedback_time)
                or not 0 <= now - feedback_time <= self.feedback_max_age_s
            ):
                raise GateRejected("feedback_stale")
            q = self.validate_q(measured_q)
            self.token = secrets.token_urlsafe(32)
            self.owner = owner
            self.state = "ACTIVE"
            self.last_time = now
            self.deadline = now + self.lease_s
            self.last_q = q
            self.last_command_time = now
            self.sequence = -1
            return self.token

    def check_session(self, token, now, selected_input, unexpected_publishers):
        if self.state != "ACTIVE":
            raise GateRejected("session_not_active")
        # Unrelated clients cannot invalidate someone else's lease with a bad token.
        if token != self.token:
            raise GateRejected("wrong_owner_token")
        self.clock(now)
        if now >= self.deadline:
            self.trip("lease_expired")
        if selected_input != 3:
            self.trip("input_selection_changed")
        if unexpected_publishers:
            self.trip("unexpected_user_publisher")

    def heartbeat(self, token, now, selected_input, unexpected_publishers=False):
        with self.lock:
            self.check_session(token, now, selected_input, unexpected_publishers)
            self.deadline = now + self.lease_s

    def accept(
        self,
        envelope,
        now,
        selected_input,
        *,
        feedback_time,
        unexpected_publishers=False,
    ):
        with self.lock:
            self.check_session(
                envelope.token, now, selected_input, unexpected_publishers
            )
            if type(envelope.sequence) is not int or envelope.sequence <= self.sequence:
                self.trip("duplicate_or_reordered_sequence")
            if (
                not math.isfinite(feedback_time)
                or not 0 <= now - feedback_time <= self.feedback_max_age_s
            ):
                self.trip("feedback_stale")
            if (
                not math.isfinite(envelope.command_time)
                or not 0 <= now - envelope.command_time <= self.command_max_age_s
            ):
                self.trip("command_stale_or_future")
            if (
                not math.isfinite(envelope.observation_time)
                or not 0 <= envelope.command_time - envelope.observation_time
                or now - envelope.observation_time > self.prediction_max_age_s
            ):
                self.trip("prediction_stale_or_future")
            if envelope.command_time <= self.last_command_time:
                self.trip("command_time_not_increasing")
            try:
                q = self.validate_q(envelope.joint_positions)
            except GateRejected as exc:
                self.trip(str(exc))
            elapsed = envelope.command_time - self.last_command_time
            delta = np.abs(q - self.last_q)
            if np.any(delta > self.velocity * elapsed + 1e-9):
                self.trip("joint_velocity")
            if np.any(delta > self.max_step):
                self.trip("joint_step")
            self.sequence = envelope.sequence
            self.last_q = q.copy()
            self.last_command_time = envelope.command_time
            # Accepting commands never renews the independent heartbeat lease.
            q.flags.writeable = False
            return q

    def poll(self, now):
        with self.lock:
            if self.state == "ACTIVE":
                self.clock(now)
                if now >= self.deadline:
                    self.state = "TRIPPED"
                    self.reason = "lease_expired"
            return {
                "state": self.state,
                "reason": self.reason,
                "stop_required": self.state == "TRIPPED",
                "hardware_stop_confirmed": False,
            }

    def finish(self, token, *, independent_stop_confirmed):
        with self.lock:
            if token != self.token or self.token is None:
                raise GateRejected("wrong_owner_token")
            if independent_stop_confirmed is not True:
                raise GateRejected("stop_not_confirmed")
            self.state = "IDLE"
            self.token = None
            self.owner = None
            self.reason = None
            self.last_time = None
            self.last_q = None
            self.last_command_time = None
            self.deadline = None
            self.sequence = -1
