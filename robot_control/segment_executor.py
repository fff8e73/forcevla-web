"""Finite joint segment executor for isolated tests only.

Keeps the solver dt, never skips/catches up ticks, never refreshes observation
age. Fixed gripper targets avoid blocking gripper transactions in the tick path.
Production use remains blocked even if a caller changes plan metadata.
"""
import copy
import math
import threading
import time
from dataclasses import dataclass

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import JointExecutionBackend
from robot_control.joint_source_guard import JointEnvelope


class BufferedSandboxBackend(JointExecutionBackend):
    """Bounded tick journal in RAM, durable at transitions/finish, sandbox only.

    A process crash can lose tick records. This is not a production logger.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if getattr(self.io, 'production', None) is not False:
            raise ControlFault('buffered_backend_requires_sandbox')

    def record(self, event, **fields):
        if event in ('submit_intent', 'transport_submission'):
            if len(self.events) >= 20000:
                raise ControlFault('tick_log_capacity')
            self.events.append(dict(time=self.clock(), event=event, state=self.state,
                                    **copy.deepcopy(fields)))
        else:
            super().record(event, **fields)


@dataclass(frozen=True)
class SegmentLimits:
    max_duration_s: float = 0.5
    max_displacement_rad: float = 0.003
    max_speed_rad_s: float = 0.02
    max_acceleration_rad_s2: float = 2.0
    max_abnormal_hold_acceleration_rad_s2: float = 10.0
    max_lateness_s: float = 0.0015
    max_observation_age_s: float = 2.5
    seed_error_rad: float = 0.001
    max_ticks: int = 1000

    def __post_init__(self):
        for key, value in vars(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError('Invalid segment limit:' + key)
        if type(self.max_ticks) is not int:
            raise ValueError('max_ticks must be an integer')


class SegmentExecutor:
    def __init__(self, backend, limits=None, *, clock=time.monotonic):
        if getattr(backend.io, 'production', None) is not False:
            raise ControlFault('segment_executor_sandbox_only')
        self.backend = backend
        self.clock = clock
        self.limits = limits or SegmentLimits()
        self.state = 'IDLE'
        self.abort_event = threading.Event()
        self.reason = None
        self.index = 0
        self.trace = []
        self.lock = threading.RLock()
        self.request_id = None
        self.plan = None

    def start(self, plan, token):
        with self.lock, self.backend.lock:
            if self.state != 'IDLE':
                raise ControlFault('executor_single_use')
            self.backend.check_token(token)
            if self.backend.state != 'ACTIVE':
                raise ControlFault('backend_not_active')
            # Freeze caller-owned data before validation to avoid later mutation.
            p = copy.deepcopy(plan)
            q, seed, grip = (np.array(v, dtype=float, copy=True)
                             for v in (p.joints, p.seed, p.grippers))
            n = len(q)
            lim = self.limits
            if (q.shape != (n, 14) or seed.shape != (14,) or grip.shape != (n, 2)
                    or not 1 <= n <= lim.max_ticks
                    or not all(np.isfinite(v).all() for v in (q, seed, grip))):
                raise ControlFault('invalid_segment_arrays')
            if (not math.isfinite(p.dt) or not 0 < p.dt <= 1/30
                    or lim.max_lateness_s >= p.dt or n*p.dt > lim.max_duration_s):
                raise ControlFault('invalid_segment_timing')
            now = self.clock()
            if (not p.request_id or not math.isfinite(p.observed_at)
                    or not 0 <= now-p.observed_at <= lim.max_observation_age_s
                    or now+n*p.dt-p.observed_at > lim.max_observation_age_s):
                raise ControlFault('segment_observation_expired')
            f = self.backend.checked_feedback(owned=True)
            if (np.max(np.abs(seed-f.joints)) > lim.seed_error_rad
                    or np.max(np.abs(seed-self.backend.last_q)) > lim.seed_error_rad):
                raise ControlFault('segment_seed_mismatch')
            if np.any(np.abs(f.velocities) > self.backend.stop_speed):
                raise ControlFault('segment_start_not_stationary')
            if (np.any(grip < 0) or np.any(grip > 1)
                    or not np.array_equal(grip, np.tile(self.backend.grip_target, (n, 1)))):
                raise ControlFault('segment_requires_fixed_grippers')
            if np.any(np.abs(q-seed) > lim.max_displacement_rad):
                raise ControlFault('segment_displacement')
            for row in q:
                self.backend.guard.validate_q(row)
            # Check entry and terminal hold; don't silently truncate a fast tail.
            velocity = np.diff(np.vstack((seed, q)), axis=0)/p.dt
            acceleration = np.diff(np.vstack((np.zeros((1,14)), velocity, np.zeros((1,14)))), axis=0)/p.dt
            if np.any(np.abs(velocity) > np.minimum(lim.max_speed_rad_s, self.backend.guard.velocity)):
                raise ControlFault('segment_speed')
            if np.any(np.abs(acceleration) > lim.max_acceleration_rad_s2):
                raise ControlFault('segment_acceleration_or_terminal_hold')
            # Every intermediate interruption can freeze the last submitted target.
            # This bounds command changes only, not actual motor braking dynamics.
            if np.any(np.abs(velocity)/p.dt > lim.max_abnormal_hold_acceleration_rad_s2):
                raise ControlFault('segment_abnormal_hold_budget')
            self.plan = (q, grip, float(p.dt), float(p.observed_at))
            self.request_id, self.token = p.request_id, token
            self.sequence = self.backend.sequence+1
            self.reason = None
            self.index = 0
            self.backend.record('segment_armed', request_id=p.request_id,
                                observed_at=p.observed_at, dt=p.dt, ticks=n,
                                limits=vars(lim), gripper_policy='fixed',
                                joint_targets=q.tolist(), hardware_execution_verified=False)
            # Journal persistence is outside the timed motion loop.
            self.started = self.clock()
            self.next_due = self.started+p.dt
            self.state = 'RUNNING'

    def request_abort(self):
        self.abort_event.set()

    def stop(self, reason):
        self.reason = reason
        self.state = 'HOLDING'
        self.plan = None  # discard all unconsumed targets
        if self.backend.state == 'ACTIVE':
            self.backend.trip(reason)
        elif self.backend.stop_report:
            self.reason = self.backend.stop_report['reason']
        self.state = self.backend.state

    def tick(self):
        with self.lock:
            if self.state != 'RUNNING':
                return self.state
            try:
                if self.abort_event.is_set():
                    self.stop('manual_abort')
                    return self.state
                if self.backend.state != 'ACTIVE':
                    self.reason = 'backend_not_active'
                    self.state = self.backend.state
                    self.plan = None
                    return self.state
                now = self.clock()
                q, grip, dt, observed = self.plan
                if now < self.started:
                    raise ControlFault('segment_clock_regression')
                if now-observed > self.limits.max_observation_age_s:
                    raise ControlFault('segment_observation_expired')
                if now < self.next_due:
                    return self.state
                lateness = now-self.next_due
                if lateness > self.limits.max_lateness_s:
                    raise ControlFault('segment_deadline_missed')
                # write() checks ownership/tracking itself; avoid a second ROS graph
                # and feedback read on the same tick. Lease also guards terminal hold.
                if self.backend.guard.poll(now)['stop_required']:
                    raise ControlFault('upstream_lease_expired')
                if self.index == len(q):
                    self.stop('segment_complete')
                    return self.state
                # Command timestamps are original fixed deadlines, never a catch-up time.
                envelope = JointEnvelope(self.token, self.sequence+self.index,
                                         self.next_due, observed, q[self.index].copy())
                begun = self.clock()
                receipts = self.backend.write(envelope, grip[self.index])
                done = self.clock()
                self.trace.append(dict(index=self.index, scheduled=self.next_due,
                                       began=begun, finished=done, lateness_s=lateness,
                                       observation_age_s=done-observed, receipts=receipts))
                if receipts != {'left':True, 'right':True, 'grippers':True}:
                    raise ControlFault('segment_partial_submission')
                self.index += 1
                self.next_due = self.started+(self.index+1)*dt
                if done-self.trace[-1]['scheduled'] > self.limits.max_lateness_s:
                    raise ControlFault('segment_submission_overrun')
            except Exception as exc:
                self.stop(str(exc))
            return self.state

    def poll_hold(self):
        with self.lock:
            if self.state == 'RUNNING':
                raise ControlFault('still_running')
            self.state = self.backend.poll()
            return self.state

    def report(self):
        return dict(state=self.state, reason=self.reason, request_id=self.request_id,
                    sent_ticks=self.index, trace=copy.deepcopy(self.trace),
                    stop_report=copy.deepcopy(self.backend.stop_report),
                    hardware_execution_verified=False, fixed_grippers=True,
                    limits=vars(self.limits))
