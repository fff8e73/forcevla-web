"""Stage B joint transport orchestration. No implicit arming or prediction replay.

A submitted ROS message is NOT a motor acknowledgement. Stationary hold is
confirmed from post-request samples; moving stop/independent watchdog remain
unverified. All transport operations must be bounded by the transport adapter.
"""

import json
import math
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_source_guard import JointSourceGuard


@dataclass(frozen=True)
class ExecutionFeedback:
    received_at: float
    source_age_s: float
    joints: np.ndarray
    velocities: np.ndarray
    joint_target: np.ndarray
    target_received_at: float
    selected_input: int
    healthy: bool
    mode: int
    gripper: dict
    gripper_received_at: float
    unexpected_publishers: bool = False


class JointExecutionBackend:
    """Explicit session backend for prepared joint commands, separate from TCP bridge.

    `io` implements feedback/select_input/publish_arm/gripper_command/heartbeat.
    Live IO is deliberately stationary-only until D/E have been implemented and
    validated. Sandbox IO can exercise arbitrary bounded synthetic commands.
    """

    def __init__(
        self,
        io,
        bounds,
        velocities,
        journal,
        *,
        clock=time.monotonic,
        feedback_age_s=0.3,
        lease_s=0.25,
        tracking_rad=0.005,
        stop_speed_rad_s=0.02,
        settle_s=0.2,
        stop_timeout_s=1.5,
        prediction_age_s=2.5,
        gripper_stop_speed_rad_s=None,
    ):
        self.io, self.clock = io, clock
        self.guard = JointSourceGuard(
            bounds,
            velocities,
            lease_s=lease_s,
            feedback_max_age_s=feedback_age_s,
            prediction_max_age_s=prediction_age_s,
        )
        for value in (tracking_rad, stop_speed_rad_s, settle_s, stop_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Invalid execution budget")
        self.feedback_age = feedback_age_s
        self.tracking = tracking_rad
        self.stop_speed = stop_speed_rad_s
        self.gripper_stop_speed = stop_speed_rad_s if gripper_stop_speed_rad_s is None else gripper_stop_speed_rad_s
        if not np.isfinite(self.gripper_stop_speed) or not 0 < self.gripper_stop_speed <= .03:
            raise ValueError('Invalid gripper stationary speed')
        self.settle_s, self.stop_timeout = settle_s, stop_timeout_s
        self.journal = Path(journal)
        self.lock = threading.RLock()
        self.state = "REVIEW_REQUIRED" if self.journal.exists() else "IDLE"
        self.token = self.boot = self.session = None
        self.initial_mode = None
        self.last_q = self.grip_target = None
        self.claim_attempted = self.selection_attempted = False
        self.events = []
        self.stop_report = None
        self.sequence = -1
        self.grip_seq = self.beat_seq = 0
        self.watchdog_stop = threading.Event()
        self.thread = None

    def record(self, event, **fields):
        self.events.append(
            dict(time=self.clock(), event=event, state=self.state, **fields)
        )
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        temp = self.journal.with_suffix(".tmp")
        # Persistent ownership intent survives process death. No auto-resume.
        with temp.open("w") as stream:
            os.chmod(temp, 0o600)
            json.dump(
                {
                    "state": self.state,
                    "boot": self.boot,
                    "session": self.session,
                    "events": self.events,
                    "stop_report": self.stop_report,
                },
                stream,
                allow_nan=False,
            )
            stream.flush()
            os.fsync(stream.fileno())
        temp.replace(self.journal)

    def record_fault(self, event, **fields):
        # Disk errors must not suppress independently possible hold requests.
        try:
            self.record(event, **fields)
        except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
            self.journal_error = repr(exc)

    def owned_gripper(self, g):
        return (
            g.get("reserved") is True
            and g.get("boot_id") == self.boot
            and g.get("session_id") == self.session
        )

    def checked_feedback(self, *, owned=False):
        f = self.io.feedback()
        now = self.clock()
        arrays = (f.joints, f.velocities, f.joint_target)
        if getattr(self, 'measured_hold', False) and f.target_received_at == 0:
            # Explicit missing-command telemetry, never relabel measured q as readback.
            arrays = (f.joints, f.velocities)
        if any(
            np.asarray(a).shape != (14,) or not np.isfinite(a).all() for a in arrays
        ):
            raise ControlFault("invalid_joint_feedback")
        if (
            not math.isfinite(f.received_at)
            or not 0 <= now - f.received_at <= self.feedback_age
            or not math.isfinite(f.source_age_s)
            or not 0 <= f.source_age_s <= self.feedback_age
        ):
            raise ControlFault("stale_joint_feedback")
        if not f.healthy or f.mode not in (1, 3):
            raise ControlFault("robot_unhealthy_or_mode")
        if self.initial_mode is not None and f.mode != self.initial_mode:
            raise ControlFault("robot_mode_changed")
        if f.unexpected_publishers:
            raise ControlFault("unexpected_user_publisher")
        g = f.gripper
        if (
            not math.isfinite(f.gripper_received_at)
            or not 0 <= now - f.gripper_received_at <= self.feedback_age
            or g.get("healthy") is not True
        ):
            raise ControlFault("gripper_unhealthy_or_stale")
        for key in (
            "actual",
            "applied_target",
            "motor_feedback_age",
            "motor_velocities",
        ):
            v = np.asarray(g.get(key), dtype=float)
            if v.shape != (2,) or not np.isfinite(v).all():
                raise ControlFault("invalid_gripper_feedback")
        ages = np.asarray(g["motor_feedback_age"]) + now - f.gripper_received_at
        if np.any(ages < 0) or np.any(ages > self.feedback_age):
            raise ControlFault("stale_motor_feedback")
        if np.any(np.asarray(g["applied_target"]) < 0) or np.any(
            np.asarray(g["applied_target"]) > 1
        ):
            raise ControlFault("invalid_existing_gripper_target")
        calibration = np.asarray(g.get("calibration"), dtype=float)
        if (
            calibration.shape != (2, 2)
            or not np.isfinite(calibration).all()
            or np.any(calibration[:, 1] <= calibration[:, 0])
        ):
            raise ControlFault("invalid_gripper_calibration")
        if owned and (not self.owned_gripper(g) or f.selected_input != 3):
            raise ControlFault("ownership_lost")
        return f

    def grip(self, action, **fields):
        self.grip_seq += 1
        request = dict(
            action=action,
            boot_id=self.boot,
            session_id=self.session,
            request_id=str(uuid.uuid4()),
            seq=self.grip_seq,
            separate_heartbeat=True,
            **fields,
        )
        if action == "hold":
            self.record_fault("gripper_request", action=action, sequence=self.grip_seq)
        else:
            self.record("gripper_request", action=action, sequence=self.grip_seq)
        result = self.io.gripper_command(request)
        if (
            result.get("last_request_id") != request["request_id"]
            or result.get("result", {}).get("ok") is not True
        ):
            raise ControlFault("gripper_receipt_mismatch_or_rejected")
        if action != "release" and not self.owned_gripper(result):
            raise ControlFault("gripper_ownership_not_confirmed")
        return result

    def acquire(self, owner):
        with self.lock:
            if self.state != "IDLE":
                raise ControlFault("session_requires_review_or_already_used")
            if getattr(self.io, "production", False) and self.thread is None:
                raise ControlFault("production_watchdog_not_running")
            f = self.checked_feedback()
            if f.selected_input != 0 or f.gripper.get("reserved") is not False:
                raise ControlFault("requires_input_none_and_free_gripper")
            if np.any(np.abs(f.velocities) > self.stop_speed):
                raise ControlFault("arms_not_stationary")
            if not 0 <= self.clock() - f.target_received_at <= self.feedback_age:
                raise ControlFault("existing_joint_target_stale")
            if np.any(np.abs(f.joints - f.joint_target) > self.tracking):
                raise ControlFault("existing_target_tracking_error")
            self.last_q = self.guard.validate_q(f.joint_target)
            self.grip_target = np.array(f.gripper["applied_target"], copy=True)
            self.initial_mode = f.mode
            self.boot, self.session = f.gripper["boot_id"], str(uuid.uuid4())
            self.state = "ACQUIRING"
            self.record(
                "acquire_intent",
                owner=owner,
                seed=self.last_q.tolist(),
                gripper_seed=self.grip_target.tolist(),
            )
            try:
                # Claim intent is durable BEFORE transmission; timeout is ambiguous.
                self.claim_attempted = True
                g = self.grip("claim")
                if not np.allclose(
                    g["applied_target"], self.grip_target, atol=1e-9, rtol=0
                ):
                    raise ControlFault("claim_changed_target")
                self.io.bind_stationary_target(self.last_q, self.grip_target)
                self.selection_attempted = True
                self.record("select_user_intent")
                self.io.select_input(3)
                f = self.checked_feedback(owned=True)
                self.token = self.guard.claim(
                    owner, self.clock(), self.last_q, f.received_at, f.selected_input
                )
                self.state = "ACTIVE"
                self.record("acquired")
                return self.token
            except Exception as exc:
                self.trip("acquire_failed:" + str(exc))
                raise

    def check_token(self, token):
        if self.token is None or token != self.token:
            raise ControlFault("wrong_owner_token")

    def heartbeat(self, token):
        with self.lock:
            self.check_token(token)
            if self.state != "ACTIVE":
                raise ControlFault("session_not_active")
            try:
                f = self.checked_feedback(owned=True)
                if f.gripper.get("state") != "active":
                    raise ControlFault("gripper_not_active_no_auto_resume")
                self.guard.heartbeat(
                    token, self.clock(), f.selected_input, f.unexpected_publishers
                )
                self.beat_seq += 1
                self.io.heartbeat(
                    {
                        "boot_id": self.boot,
                        "session_id": self.session,
                        "beat_seq": self.beat_seq,
                    }
                )
            except Exception as exc:
                self.trip(str(exc))
                raise

    def write(self, envelope, grippers=None):
        """One caller-scheduled joint tick. Never renews the heartbeat lease.

        Returns transport submission receipts, NOT execution acknowledgements.
        Gripper changes use bounded acknowledged requests; this synchronous B
        interface is not suitable for a 500 Hz moving trajectory scheduler.
        """
        with self.lock:
            self.check_token(envelope.token)
            if self.state != "ACTIVE":
                raise ControlFault("session_not_active")
            try:
                f = self.checked_feedback(owned=True)
                if f.gripper.get("state") != "active":
                    raise ControlFault("gripper_not_active_no_auto_resume")
                if np.any(np.abs(f.joints - self.last_q) > self.tracking):
                    raise ControlFault("joint_tracking_error")
                q = self.guard.accept(
                    envelope,
                    self.clock(),
                    f.selected_input,
                    feedback_time=f.received_at,
                    unexpected_publishers=f.unexpected_publishers,
                )
                grip = (
                    self.grip_target
                    if grippers is None
                    else np.asarray(grippers, float)
                )
                if (
                    grip.shape != (2,)
                    or not np.isfinite(grip).all()
                    or np.any(grip < 0)
                    or np.any(grip > 1)
                ):
                    raise ControlFault("invalid_gripper_target")
                # Adapter must validate both arms and grip before any partial send.
                self.io.validate_targets(q, grip)
                self.record(
                    "submit_intent",
                    sequence=envelope.sequence,
                    observation_time=envelope.observation_time,
                    command_time=envelope.command_time,
                    joints=q.tolist(),
                    grippers=grip.tolist(),
                )
                submitted = {"left": False, "right": False, "grippers": False}
                try:
                    for side, index in [("left", 0), ("right", 7)]:
                        self.io.publish_arm(side, q[index : index + 7])
                        submitted[side] = True
                        self.last_q[index : index + 7] = q[index : index + 7]
                    for index, side in enumerate(("left", "right")):
                        if grip[index] != self.grip_target[index]:
                            self.grip(
                                "target",
                                side=side,
                                value=float(grip[index]),
                                mode="fine",
                                target_generation=f.gripper["target_generation"],
                            )
                    submitted["grippers"] = True
                finally:
                    self.record(
                        "transport_submission",
                        sequence=envelope.sequence,
                        submitted=submitted,
                        execution_acknowledged=False,
                    )
                self.last_q, self.grip_target = q.copy(), grip.copy()
                self.sequence = envelope.sequence
                return submitted
            except Exception as exc:
                self.trip(str(exc))
                raise

    def trip(self, reason):
        with self.lock:
            if self.state in (
                "HOLD_CONFIRMED",
                "FAULT_UNCONFIRMED",
                "RELEASED",
                "REVIEW_REQUIRED",
            ):
                return self.stop_report
            self.state = "STOPPING"
            self.record_fault("fault_latched", reason=reason)
            report = {
                "reason": reason,
                "arms_hold_sent": False,
                "gripper_held": False,
                "stationary_feedback_confirmed": False,
                "hardware_stop_verified": False,
                "errors": [],
            }
            # Ownership is checked afresh. Never write into a third party's session.
            try:
                f = self.checked_feedback()
            except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                f = None
                report["errors"].append("feedback:" + str(exc))
            if (
                f is not None
                and self.selection_attempted
                and f.selected_input == 3
                and not f.unexpected_publishers
            ):
                try:
                    # Stage B holds the last submitted target, not a guessed brake trajectory.
                    self.io.validate_targets(self.last_q, self.grip_target)
                    sent = {}
                    for side, index in [("left", 0), ("right", 7)]:
                        try:
                            self.io.publish_arm(side, self.last_q[index : index + 7])
                            sent[side] = True
                        except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                            sent[side] = False
                            report["errors"].append(side + "_hold:" + str(exc))
                    report["arm_hold_submissions"] = sent
                    report["arms_hold_sent"] = all(sent.values())
                except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                    report["errors"].append("arms_hold:" + str(exc))
            # Try the gripper independently even if arm status/readback failed.
            try:
                g = self.io.gripper_status()
                if self.claim_attempted and self.owned_gripper(g):
                    held = self.grip("hold")
                    report["gripper_held"] = held.get("state") == "held"
                    if report["gripper_held"]:
                        self.grip_target = np.array(held["applied_target"], dtype=float)
            except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                report["errors"].append("gripper_hold:" + str(exc))
            self.stop_report = report
            self.stop_started = self.clock()
            self.settled_since = None
            self.last_stop_sample = None
            if not report["arms_hold_sent"] or not report["gripper_held"]:
                self.state = "FAULT_UNCONFIRMED"
            self.record_fault("hold_requested", report=report)
            return report

    def poll(self):
        with self.lock:
            if self.state == "ACTIVE":
                try:
                    f = self.checked_feedback(owned=True)
                    if f.gripper.get("state") != "active":
                        raise ControlFault("gripper_held_or_faulted")
                    if np.any(np.abs(f.joints - self.last_q) > self.tracking):
                        raise ControlFault("joint_tracking_error")
                    if self.guard.poll(self.clock())["stop_required"]:
                        raise ControlFault("upstream_lease_expired")
                except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                    self.trip(str(exc))
            if self.state != "STOPPING":
                return self.state
            try:
                f = self.checked_feedback(owned=True)
                self.io.validate_targets(self.last_q, self.grip_target)
                for side, index in [("left", 0), ("right", 7)]:
                    self.io.publish_arm(side, self.last_q[index : index + 7])
                if (
                    f.received_at <= self.stop_started
                    or f.gripper_received_at <= self.stop_started
                    or f.target_received_at <= self.stop_started
                ):
                    raise ControlFault("awaiting_post_hold_feedback")
                if (
                    f.gripper.get("state") != "held"
                    or np.any(np.abs(f.velocities) > self.stop_speed)
                    or np.any(np.abs(f.joints - self.last_q) > self.tracking)
                    or not np.allclose(f.joint_target, self.last_q, atol=1e-8, rtol=0)
                    or not np.allclose(
                        f.gripper["applied_target"], self.grip_target, atol=1e-8, rtol=0
                    )
                ):
                    raise ControlFault("hold_not_settled")
                # Motor velocities come from independently subscribed feedback.
                v = np.asarray(f.gripper.get("motor_velocities"), dtype=float)
                if (
                    v.shape != (2,)
                    or not np.isfinite(v).all()
                    or np.any(np.abs(v) > self.gripper_stop_speed)
                ):
                    raise ControlFault("gripper_not_stationary")
                calibration = np.asarray(f.gripper["calibration"])
                grip_error = np.abs(
                    np.asarray(f.gripper["actual"]) - self.grip_target
                ) * (calibration[:, 1] - calibration[:, 0])
                if np.any(grip_error > 0.05):
                    raise ControlFault("gripper_hold_tracking_error")
                sample = min(f.received_at, f.gripper_received_at, f.target_received_at)
                if self.last_stop_sample is None or sample > self.last_stop_sample:
                    self.last_stop_sample = sample
                    if self.settled_since is None:
                        self.settled_since = sample
                    if sample - self.settled_since >= self.settle_s:
                        self.state = (
                            "FAULT_UNCONFIRMED"
                            if hasattr(self, "journal_error")
                            else "HOLD_CONFIRMED"
                        )
                        self.stop_report["hold_feedback_confirmed"] = True
                        self.stop_report["joint_stationary_feedback_confirmed"] = True
                        self.stop_report["stationary_feedback_confirmed"] = bool(self.gripper_stop_speed <= .03 and np.all(np.abs(v) <= .03))
                        self.stop_report["gripper_speed_monitor_limit_rad_s"] = self.gripper_stop_speed
                        self.stop_report["feedback_evidence"] = {
                            "received_at": f.received_at,
                            "sample_span_s": sample - self.settled_since,
                            "max_joint_speed": float(np.max(np.abs(f.velocities))),
                            "max_joint_error": float(
                                np.max(np.abs(f.joints - self.last_q))
                            ),
                            "gripper_speed": v.tolist(),
                            "gripper_error_rad": grip_error.tolist(),
                        }
                        self.record_fault("stationary_hold_confirmed")
            except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                self.settled_since = None
                self.last_stop_sample = None
                self.stop_report["last_feedback_issue"] = str(exc)
            if (
                self.state == "STOPPING"
                and self.clock() - self.stop_started >= self.stop_timeout
            ):
                self.state = "FAULT_UNCONFIRMED"
                self.record_fault("hold_confirmation_timeout")
            return self.state

    def release(self, token):
        with self.lock:
            self.check_token(token)
            if self.state != "HOLD_CONFIRMED":
                raise ControlFault("hold_not_confirmed_keep_ownership")
            try:
                f = self.checked_feedback(owned=True)
                if (
                    np.any(np.abs(f.velocities) > self.stop_speed)
                    or np.any(np.abs(f.joints - self.last_q) > self.tracking)
                    or np.any(np.abs(f.gripper["motor_velocities"]) > self.gripper_stop_speed)
                    or f.gripper.get("state") != "held"
                    or not np.allclose(
                        f.gripper["applied_target"], self.grip_target, atol=1e-8, rtol=0
                    )
                ):
                    raise ControlFault("release_feedback_changed")
                self.record("select_none_intent")
                self.io.select_input(0)
                if self.io.feedback().selected_input != 0:
                    raise ControlFault("input_none_unconfirmed")
                g = self.grip("release")
                if g.get("reserved") is not False or g.get("session_id") is not None:
                    raise ControlFault("release_unconfirmed")
                self.state = "RELEASED"
                self.record("released_input_none")
                # Keep terminal journal as evidence, new session needs a new path.
            except Exception as exc:
                self.state = "FAULT_UNCONFIRMED"
                self.record("release_unconfirmed", error=str(exc))
                raise

    def start_watchdog(self, period_s=0.02):
        if not 0 < period_s <= 0.05 or self.thread is not None:
            raise ValueError("Invalid watchdog start")

        def run():
            while not self.watchdog_stop.wait(period_s):
                try:
                    self.poll()
                except Exception as exc:  # noqa: BLE001 - preserve all IO failures at this boundary
                    # An IO/journal failure must not silently leave ACTIVE.
                    self.state = "FAULT_UNCONFIRMED"
                    self.stop_report = {
                        "error": repr(exc),
                        "hardware_stop_verified": False,
                    }

        self.thread = threading.Thread(
            target=run, name="execution-watchdog", daemon=True
        )
        self.thread.start()

    def close(self):
        if self.state == "ACTIVE":
            self.trip("backend_closed")
        self.watchdog_stop.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.state == "STOPPING":
            self.state = "FAULT_UNCONFIRMED"
            self.record_fault("closed_before_hold_confirmation")
        # Never auto-release an unconfirmed session on shutdown.
        self.io.close()
