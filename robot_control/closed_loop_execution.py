"""Reusable execution session: normal holds preserve ownership; stops latch.

The controller thread only uses bounded cached IO, in-memory plans and a bounded
journal queue. RPC/IK never run here. Production qualification is a transport
responsibility; no sandbox flag is changed to enable hardware.
"""

import copy
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path

import numpy as np

from robot_control.closed_loop_contracts import Barrier
from robot_control.closed_loop_planning import kinematic_checks
from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import JointExecutionBackend
from robot_control.joint_source_guard import JointEnvelope


class BoundedJournal:
    def __init__(self, folder, limit_mb):
        self.path = Path(folder)
        self.path.mkdir(parents=True, exist_ok=True)
        self.queue = queue.Queue(maxsize=8192)
        self.error = None
        self.limit = limit_mb * 1024 * 1024
        self.bytes = 0
        self.finished = threading.Event()
        self.thread = threading.Thread(
            target=self._run, name="closed-loop-journal", daemon=True
        )
        self.thread.start()

    def append(self, event):
        if self.error:
            raise ControlFault("journal_failure:" + self.error)
        try:
            self.queue.put_nowait(copy.deepcopy(event))
        except queue.Full:
            raise ControlFault("journal_queue_full") from None

    def _run(self):
        try:
            with (self.path / "execution.jsonl").open("x") as stream:
                while not self.finished.is_set() or not self.queue.empty():
                    try:
                        row = self.queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    data = json.dumps(row, allow_nan=False) + "\n"
                    self.bytes += len(data.encode())
                    if self.bytes > self.limit:
                        raise ControlFault("journal_quota")
                    stream.write(data)
                    if self.queue.empty():
                        stream.flush()
                stream.flush()
                os.fsync(stream.fileno())
        except Exception as exc:
            self.error = repr(exc)

    def close(self):
        self.finished.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            self.error = "journal_shutdown_timeout"
        if self.error:
            raise ControlFault(self.error)


class LoopBackend(JointExecutionBackend):
    measured_hold = True

    def __init__(self, io, bounds, config, journal):
        self.audit = journal
        self.gripper_serial = threading.Lock()
        super().__init__(
            io,
            bounds,
            np.full(14, config.max_joint_speed_rad_s),
            journal.path / "ownership.json",
            feedback_age_s=config.feedback_age_s,
            lease_s=config.supervision_timeout_s,
            tracking_rad=config.tracking_error_rad,
            stop_speed_rad_s=config.stop_speed_rad_s,
            gripper_stop_speed_rad_s=config.gripper_stop_speed_rad_s,
            settle_s=config.settle_time_s,
            stop_timeout_s=config.hold_timeout_s,
            prediction_age_s=config.max_prediction_age_s,
        )

    def record(self, event, **fields):
        row = dict(time=self.clock(), event=event, state=self.state, **fields)
        # Durable acquisition/release intent is outside the timed execution path.
        if event in (
            "acquire_intent",
            "select_user_intent",
            "select_none_intent",
            "released_input_none",
        ):
            super().record(event, **fields)
        self.audit.append(row)

    def grip(self, action, **fields):
        with self.gripper_serial:
            return super().grip(action, **fields)

    def acquire(self, owner):
        with self.lock:
            if self.state != "IDLE":
                raise ControlFault("session_not_idle")
            f = self.checked_feedback()
            if (
                f.mode != 1
                or f.selected_input != 0
                or f.gripper.get("reserved") is not False
            ):
                raise ControlFault("requires_position_none_free_gripper")
            if (
                np.max(np.abs(f.velocities)) > self.stop_speed
                or np.max(np.abs(f.gripper["motor_velocities"]))
                > self.gripper_stop_speed
            ):
                raise ControlFault("acquire_not_stationary")
            self.last_q = self.guard.validate_q(f.joints)
            self.grip_target = np.array(
                f.gripper["applied_target"], dtype=float, copy=True
            )
            self.initial_mode = f.mode
            self.boot, self.session = f.gripper["boot_id"], str(uuid.uuid4())
            self.state = "ACQUIRING"
            self.record(
                "acquire_intent",
                owner=owner,
                seed=self.last_q.tolist(),
                gripper_seed=self.grip_target.tolist(),
                policy="fixed_measured_seed",
            )
            try:
                self.claim_attempted = True
                result = self.grip("claim")
                if not np.allclose(
                    result["applied_target"], self.grip_target, atol=1e-9, rtol=0
                ):
                    raise ControlFault("claim_changed_target")
                current = self.checked_feedback()
                if (
                    np.max(np.abs(current.joints - self.last_q)) > self.tracking
                    or np.max(np.abs(current.velocities)) > self.stop_speed
                ):
                    raise ControlFault("moved_during_claim")
                self.io.bind_stationary_target(self.last_q, self.grip_target)
                self.selection_attempted = True
                self.record("select_user_intent")
                self.io.select_input(3)
                f = self.checked_feedback(owned=True)
                self.token = self.guard.claim(
                    owner, self.clock(), self.last_q, f.received_at, 3
                )
                self.state = "ACTIVE"
                self.record("acquired", session=self.session)
                return self.token
            except Exception as exc:
                self.trip("acquire_failed:" + str(exc))
                raise


class ContinuousSession:
    def __init__(self, io, bounds, config, folder):
        if not getattr(io, "bounded_cached_feedback", False):
            raise ControlFault("execution_requires_bounded_cached_io")
        self.io, self.config = io, config
        self.audit = BoundedJournal(folder, config.max_log_mb)
        self.backend = LoopBackend(io, bounds, config, self.audit)
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.phase = "IDLE"
        self.barrier = None
        self.segment = None
        self.index = self.executed_steps = self.submitted_ticks = 0
        self.seen_plans = set()
        self.cancellation = None
        self.stop_reason = None
        self.stop_result = None
        self.terminal_thread = None
        self.last_supervision = time.monotonic()
        self.last_beat = 0.0
        self.next_due = 0.0
        self.settled_since = self.last_settle_sample = None
        self.settle_started = 0.0
        self.initial_settling = False
        self.last_velocity = np.zeros(14)
        self.gripper_queue = queue.Queue(maxsize=1)
        self.gripper_pending = False
        self.gripper_error = None
        self.finished = threading.Event()
        self.gripper_thread = threading.Thread(
            target=self._gripper_loop, name="loop-gripper", daemon=True
        )
        self.controller_thread = threading.Thread(
            target=self._run, name="loop-controller", daemon=True
        )
        self.gripper_thread.start()
        self.controller_thread.start()

    def supervise(self):
        self.last_supervision = time.monotonic()

    def acquire(self, owner):
        with self.changed:
            if self.phase != "IDLE":
                raise ControlFault("session_not_idle")
            self.phase = "ACQUIRING"
        self.backend.acquire(owner)
        with self.changed:
            if self.stop_reason is not None or self.phase != "ACQUIRING":
                raise ControlFault("acquisition_cancelled")
            self._settling(initial=True)
            self.changed.notify_all()
        return self.wait_hold(None)

    def _settling(self, initial=False):
        self.initial_settling = initial and getattr(self.io, "has_source_ramp", False)
        self.phase = "SETTLING"
        self.settle_started = time.monotonic()
        self.settled_since = self.last_settle_sample = None
        self.next_due = self.settle_started

    def wait_hold(self, cancellation):
        end = time.monotonic() + (
            self.config.acquire_timeout_s
            if self.initial_settling
            else self.config.hold_timeout_s
        )
        with self.changed:
            while self.phase != "HOLDING":
                if cancellation:
                    cancellation.check()
                if self.phase in (
                    "STOPPING",
                    "STOP_CONFIRMED",
                    "STOP_UNCONFIRMED",
                    "RELEASED",
                ):
                    raise ControlFault(self.stop_reason or "session_stopped")
                if time.monotonic() >= end:
                    raise ControlFault("normal_hold_timeout")
                self.changed.wait(0.01)
            return self.barrier

    def execute(self, segment, cancellation):
        p = copy.deepcopy(segment)
        c = self.config
        now = time.monotonic()
        failures, _, velocity = kinematic_checks(
            p.joints, p.seed, p.seed_velocity, c, self.backend.guard.bounds
        )
        if (
            failures
            or p.checks.get("status") != "prepared"
            or p.checks.get("config_sha256") != c.digest()
        ):
            raise ControlFault("segment_not_prepared_for_profile")
        if p.dt != c.control_period_s or len(p.brake_paths) != len(p.joints):
            raise ControlFault("invalid_segment_timing_or_brakes")
        if (
            not 0 <= now - p.observed_at
            or now + p.duration_s + p.checks["max_brake_s"] - p.observed_at
            > c.max_prediction_age_s
        ):
            raise ControlFault("segment_age_budget")
        if (
            not np.isfinite(p.grippers).all()
            or np.any(p.grippers < 0)
            or np.any(p.grippers > 1)
        ):
            raise ControlFault("invalid_segment_grippers")
        cancellation.check(p.identity.generation)
        with self.changed, self.backend.lock:
            if self.phase != "HOLDING" or self.backend.state != "ACTIVE":
                raise ControlFault("execute_requires_normal_hold")
            f = self.backend.checked_feedback(owned=True)
            if (
                np.max(np.abs(f.joints - p.seed)) > c.seed_error_rad
                or np.max(np.abs(self.backend.last_q - p.seed)) > c.seed_error_rad
                or np.max(np.abs(f.velocities - p.seed_velocity)) > c.stop_speed_rad_s
            ):
                raise ControlFault("execution_seed_changed")
            if not np.allclose(
                self.backend.grip_target, p.initial_grippers, atol=1e-8, rtol=0
            ):
                raise ControlFault("execution_gripper_seed_changed")
            plan_id = p.digest()
            if plan_id in self.seen_plans:
                raise ControlFault("plan_replay")
            self.io.arm_segment(p)
            self.seen_plans.add(plan_id)
            self.segment, self.cancellation = p, cancellation
            self.velocity = velocity
            self.index = self.executed_steps = 0
            self.last_velocity = np.array(p.seed_velocity, copy=True)
            self.started = time.monotonic()
            self.next_due = self.started + p.dt
            self.phase = "EXECUTE"
            self.audit.append(
                dict(
                    event="segment_started",
                    time=self.started,
                    plan_id=plan_id,
                    identity=asdict(p.identity),
                    observed_at=p.observed_at,
                )
            )
            self.changed.notify_all()
        with self.changed:
            while self.phase in ("EXECUTE", "SETTLING", "BRAKING"):
                self.changed.wait(0.01)
            if self.phase != "HOLDING":
                raise ControlFault(self.stop_reason or "execution_stopped")
            result = dict(
                plan_id=plan_id,
                submitted_ticks=self.index,
                executed_model_steps=self.executed_steps,
                progress_basis="submitted_control_intervals_with_feedback_checks",
                completed_mono=time.monotonic(),
                hold=asdict(self.barrier),
                hardware_execution_verified=False,
            )
            self.segment = None
            return result

    def _publish(self, q, due, *, prediction=True):
        b = self.backend
        f = b.checked_feedback(owned=True)
        if f.gripper.get("state") != "active":
            raise ControlFault("gripper_session_not_active")
        if np.max(np.abs(f.joints - b.last_q)) > self.config.tracking_error_rad:
            raise ControlFault("joint_tracking_error")
        if prediction:
            envelope = JointEnvelope(
                b.token, b.sequence + 1, due, self.segment.observed_at, q
            )
            q = b.guard.accept(
                envelope,
                time.monotonic(),
                f.selected_input,
                feedback_time=f.received_at,
                unexpected_publishers=f.unexpected_publishers,
            )
            b.sequence = envelope.sequence
        self.io.validate_targets(q, b.grip_target)
        submitted = dict(left=False, right=False)
        completed = None
        try:
            for side, offset in (("left", 0), ("right", 7)):
                self.io.publish_arm(side, q[offset : offset + 7])
                b.last_q[offset : offset + 7] = q[offset : offset + 7]
                submitted[side] = True
            completed = time.monotonic()
        finally:
            self.audit.append(
                dict(
                    event="joint_submission",
                    time=completed or time.monotonic(),
                    due=due,
                    kind="prediction" if prediction else "hold_or_brake",
                    submitted=submitted,
                    positions=b.last_q.tolist(),
                    measured_positions=f.joints.tolist(),
                    measured_velocities=f.velocities.tolist(),
                    feedback_time=f.received_at,
                    original_observation_time=self.segment.observed_at
                    if self.segment
                    else None,
                )
            )
        self.submitted_ticks += 1
        return completed

    def _gripper_loop(self):
        while not self.finished.is_set():
            try:
                values, generation = self.gripper_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                for index, side in enumerate(("left", "right")):
                    # Serialize target/hold publication, without locking the controller
                    # while waiting for a driver receipt.
                    with self.backend.gripper_serial:
                        with self.changed:
                            if (
                                self.phase != "EXECUTE"
                                or self.cancellation.event.is_set()
                                or generation != self.cancellation.generation
                            ):
                                raise ControlFault("gripper_target_cancelled")
                            if values[index] == self.backend.grip_target[index]:
                                continue
                            f = self.backend.checked_feedback(owned=True)
                            self.backend.grip_seq += 1
                            request = dict(
                                action="target",
                                boot_id=self.backend.boot,
                                session_id=self.backend.session,
                                request_id=uuid.uuid4().hex,
                                seq=self.backend.grip_seq,
                                separate_heartbeat=True,
                                side=side,
                                value=float(values[index]),
                                mode="fine",
                                target_generation=f.gripper["target_generation"],
                            )
                        result = self.io.gripper_command(request)
                        with self.changed:
                            if (
                                result.get("last_request_id") != request["request_id"]
                                or result.get("result", {}).get("ok") is not True
                                or not self.backend.owned_gripper(result)
                            ):
                                raise ControlFault("gripper_receipt_mismatch")
                            if (
                                self.phase not in ("EXECUTE", "SETTLING")
                                or self.cancellation.event.is_set()
                            ):
                                raise ControlFault("gripper_receipt_after_cancel")
                            self.backend.grip_target[index] = values[index]
                            self.audit.append(
                                dict(
                                    event="gripper_receipt",
                                    time=time.monotonic(),
                                    request=request,
                                    actual=result.get("actual"),
                                    applied_target=result.get("applied_target"),
                                )
                            )
            except Exception as exc:
                self.gripper_error = str(exc)
            finally:
                self.gripper_pending = False

    def _normal_hold(self, now):
        b, c = self.backend, self.config
        self._publish(b.last_q.copy(), now, prediction=False)
        f = b.checked_feedback(owned=True)
        if self.phase != "SETTLING":
            return
        if now - self.settle_started > (
            c.acquire_timeout_s if self.initial_settling else c.hold_timeout_s
        ):
            raise ControlFault("normal_hold_timeout")
        sample = min(f.received_at, f.gripper_received_at, f.target_received_at)
        calibration = np.asarray(f.gripper["calibration"])
        grip_error = np.abs(np.asarray(f.gripper["actual"]) - b.grip_target) * (
            calibration[:, 1] - calibration[:, 0]
        )
        settled = (
            not self.gripper_pending
            and sample > self.settle_started
            and np.max(np.abs(f.velocities)) <= c.stop_speed_rad_s
            and np.max(np.abs(f.gripper["motor_velocities"]))
            <= c.gripper_stop_speed_rad_s
            and np.max(np.abs(f.joints - b.last_q)) <= c.tracking_error_rad
            and np.max(grip_error) <= c.gripper_tracking_error_rad
            and np.allclose(f.joint_target, b.last_q, atol=1e-8, rtol=0)
            and np.allclose(
                f.gripper["applied_target"], b.grip_target, atol=1e-8, rtol=0
            )
        )
        if not settled:
            self.settled_since = self.last_settle_sample = None
        elif self.last_settle_sample is None or sample > self.last_settle_sample:
            if self.settled_since is None:
                self.settled_since = sample
            self.last_settle_sample = sample
            if sample - self.settled_since >= c.settle_time_s and (
                not self.initial_settling
                or now - self.settle_started >= c.source_ramp_s
            ):
                self.barrier = Barrier.now()
                self.phase = "HOLDING"
                self.audit.append(
                    dict(
                        event="normal_hold_confirmed",
                        time=now,
                        barrier=asdict(self.barrier),
                        sample_time=sample,
                        session=b.session,
                        joints=f.joints.tolist(),
                        gripper_actual=f.gripper["actual"],
                    )
                )
                self.changed.notify_all()

    def _start_stop(self, reason, *, allow_brake=True):
        if self.phase in (
            "BRAKING",
            "STOPPING",
            "STOP_CONFIRMED",
            "STOP_UNCONFIRMED",
            "RELEASED",
        ):
            return
        self.stop_reason = reason
        if self.cancellation:
            self.cancellation.cancel(reason)
        # Only a previously checked branch may produce stopping motion.
        if allow_brake and self.phase == "EXECUTE" and self.index > 0:
            self.braking = self.segment.brake_paths[self.index - 1]
            self.brake_index = 0
            self.phase = "BRAKING"
            self.next_due = time.monotonic() + self.config.control_period_s
            self.audit.append(
                dict(
                    event="braking_started",
                    time=time.monotonic(),
                    reason=reason,
                    index=self.index - 1,
                )
            )
        else:
            self._terminal()
        self.changed.notify_all()

    def _terminal(self):
        if self.terminal_thread is not None:
            return
        self.phase = "STOPPING"
        self.segment = None

        def terminate():
            try:
                self.backend.trip(self.stop_reason or "normal_completion")
                end = time.monotonic() + self.config.hold_timeout_s
                while self.backend.state == "STOPPING" and time.monotonic() < end:
                    self.backend.poll()
                    time.sleep(0.01)
                report = copy.deepcopy(self.backend.stop_report or {})
                confirmed = self.backend.state == "HOLD_CONFIRMED"
            except Exception as exc:
                report, confirmed = dict(error=repr(exc)), False
            with self.changed:
                self.stop_result = dict(
                    report, confirmed=confirmed, reason=self.stop_reason
                )
                self.phase = "STOP_CONFIRMED" if confirmed else "STOP_UNCONFIRMED"
                self.changed.notify_all()

        self.terminal_thread = threading.Thread(
            target=terminate, name="loop-stop", daemon=True
        )
        self.terminal_thread.start()

    def _run(self):
        while not self.finished.is_set():
            try:
                with self.changed:
                    now = time.monotonic()
                    if self.backend.state != "ACTIVE" or self.phase not in (
                        "EXECUTE",
                        "HOLDING",
                        "SETTLING",
                        "BRAKING",
                    ):
                        self.changed.wait(0.005)
                        continue
                    if self.audit.error:
                        raise ControlFault("journal_failure:" + self.audit.error)
                    if now - self.last_supervision > self.config.supervision_timeout_s:
                        self._start_stop("supervisor_expired")
                    if self.gripper_error:
                        self._start_stop("gripper:" + self.gripper_error)
                    if self.cancellation and self.cancellation.event.is_set():
                        self._start_stop(self.cancellation.reason)
                    if self.phase not in ("EXECUTE", "HOLDING", "SETTLING", "BRAKING"):
                        continue
                    if now - self.last_beat >= self.config.heartbeat_period_s:
                        # No lease extension after supervision loss. Braking/hold IO
                        # remains explicit and is never relabeled as a new prediction.
                        if (
                            now - self.last_supervision
                            <= self.config.supervision_timeout_s
                        ):
                            self.backend.heartbeat(self.backend.token)
                        self.last_beat = now
                    if now < self.next_due:
                        delay = self.next_due - now
                    else:
                        delay = 0.0
                        if self.phase == "EXECUTE":
                            p, i = self.segment, self.index
                            if now - self.next_due > self.config.max_lateness_s:
                                self.audit.append(
                                    dict(
                                        event="deadline_missed",
                                        time=now,
                                        due=self.next_due,
                                        lateness_s=now - self.next_due,
                                        index=self.index,
                                    )
                                )
                                self._start_stop("control_deadline_missed")
                                continue
                            if now - p.observed_at > self.config.max_prediction_age_s:
                                self._start_stop("prediction_expired")
                                continue
                            if i == len(p.joints):
                                self._settling()
                                continue
                            grip = p.grippers[i]
                            if i == 0 or p.target_indices[i] != p.target_indices[i - 1]:
                                if not np.array_equal(grip, self.backend.grip_target):
                                    if self.gripper_pending:
                                        raise ControlFault("gripper_lane_overrun")
                                    self.gripper_pending = True
                                    self.gripper_queue.put_nowait(
                                        (grip.copy(), p.identity.generation)
                                    )
                            submitted_at = self._publish(p.joints[i], self.next_due)
                            self.last_velocity = self.velocity[i]
                            self.index += 1
                            if (
                                self.index == len(p.joints)
                                or p.target_indices[self.index] != p.target_indices[i]
                            ):
                                self.executed_steps = int(p.target_indices[i]) + 1
                            if (
                                submitted_at - self.next_due
                                > self.config.max_lateness_s
                            ):
                                self._start_stop("submission_deadline_missed")
                                continue
                            self.next_due = self.started + (self.index + 1) * p.dt
                        elif self.phase == "BRAKING":
                            if now - self.next_due > self.config.max_lateness_s:
                                self._terminal()
                                continue
                            if self.brake_index == len(self.braking):
                                self._terminal()
                                continue
                            self._publish(
                                self.braking[self.brake_index],
                                self.next_due,
                                prediction=False,
                            )
                            self.brake_index += 1
                            self.next_due += self.config.control_period_s
                        else:
                            self._normal_hold(now)
                            self.next_due = now + 0.01
                if delay > 0.0003:
                    with self.changed:
                        remaining = self.next_due - time.monotonic()
                        if remaining > 0.0002:
                            self.changed.wait(remaining - 0.0002)
                elif delay > 0:
                    # Very short deadline finish; no sleep can exceed one tick.
                    while (
                        time.monotonic() < self.next_due and not self.finished.is_set()
                    ):
                        pass
            except Exception as exc:
                with self.changed:
                    self.stop_reason = self.stop_reason or str(exc)
                    self._start_stop(str(exc), allow_brake=False)
                    if self.phase == "BRAKING":
                        self._terminal()

    def stop(self, reason):
        with self.changed:
            if self.phase == "IDLE" and self.backend.state == "IDLE":
                self.stop_reason = reason
                self.phase = "STOP_CONFIRMED"
                self.stop_result = dict(confirmed=True, reason=reason, acquired=False)
            else:
                self._start_stop(reason)
            end = time.monotonic() + self.config.hold_timeout_s + 1
            while self.phase not in ("STOP_CONFIRMED", "STOP_UNCONFIRMED", "RELEASED"):
                if time.monotonic() > end:
                    self.phase = "STOP_UNCONFIRMED"
                    self.stop_result = dict(
                        confirmed=False, reason=reason, error="stop_timeout"
                    )
                    break
                self.changed.wait(0.01)
            return copy.deepcopy(self.stop_result)

    def release(self):
        with self.changed:
            if self.phase != "STOP_CONFIRMED":
                raise ControlFault("release_requires_confirmed_stop")
            if self.backend.token is not None:
                self.backend.release(self.backend.token)
            self.phase = "RELEASED"
            self.changed.notify_all()

    def status(self):
        with self.lock:
            return dict(
                phase=self.phase,
                session=self.backend.session,
                hold_target=self.backend.last_q.tolist()
                if self.backend.last_q is not None
                else None,
                submitted_ticks=self.submitted_ticks,
                current_segment_ticks=self.index,
                executed_model_steps=self.executed_steps,
                reason=self.stop_reason,
                hold=asdict(self.barrier) if self.barrier else None,
                stop=copy.deepcopy(self.stop_result),
                gripper_pending=self.gripper_pending,
            )

    def close(self):
        if self.phase not in ("RELEASED", "STOP_CONFIRMED", "STOP_UNCONFIRMED", "IDLE"):
            self.stop("session_closed")
        self.finished.set()
        self.controller_thread.join(timeout=2)
        self.gripper_thread.join(timeout=2)
        if self.terminal_thread:
            self.terminal_thread.join(timeout=2)
        self.io.close()
        self.audit.close()
