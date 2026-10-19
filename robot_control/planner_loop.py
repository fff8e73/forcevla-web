"""Observe -> infer -> configurable retimed MoveJ waypoints -> fresh observation.

Uses installed planner limits. Raw wrench remains diagnostic only. No saved
prediction replay and no resumption of interrupted queues. Browser recovery
always acquires a new observation and replans the interrupted cycle.
"""
import json
import math
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import save_json
from robot_control.first10_replay import ReplayRunner, check_start, prepare
from robot_control.run_first10 import PlannerIO


DEFAULTS = dict(execution_horizon=20, max_cycles=50, max_task_duration_s=600.,
                browser_disconnect_timeout_s=5., observation_timeout_s=5.,
                max_observation_age_s=.75, max_source_skew_s=.2, feedback_age_s=.75,
                clock_tolerance_s=.05, seed_error_rad=.002, rpc_timeout_s=15.,
                prepare_timeout_s=15., max_gripper_correction=.05)


def profile(values):
    c = dict(DEFAULTS, **values)
    if type(c['execution_horizon']) is not int or not 1 <= c['execution_horizon'] <= 50:
        raise ValueError('execution_horizon must be 1..50')
    if type(c['max_cycles']) is not int or not 1 <= c['max_cycles'] <= 100:
        raise ValueError('max_cycles must be 1..100')
    for key in DEFAULTS:
        value = c[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('Invalid planner loop option:' + key)
    if c['max_task_duration_s'] > 600:
        raise ValueError('Task duration must be <=600 seconds')
    return SimpleNamespace(**c)


def adapt(actions, max_correction=.05, *, execution_horizon=20):
    raw = np.asarray(actions)
    if raw.shape != (1, 50, 14) or not np.isfinite(raw).all():
        raise ControlFault('invalid_model_actions')
    if type(execution_horizon) is not int or not 1 <= execution_horizon <= 50:
        raise ValueError('execution_horizon must be 1..50')
    bounded = np.array(raw[0, :execution_horizon], dtype=float, copy=True)
    old = bounded[:, [6, 13]].copy()
    bounded[:, [6, 13]] = np.clip(old, 0, 1)
    correction = float(np.max(np.abs(bounded[:, [6, 13]] - old)))
    if correction > max_correction:
        raise ControlFault('gripper_correction_excessive')
    return bounded, correction


def prepare_waypoints(solver, actions, observation, output):
    """Sequential batches of up to five goals; validate the selected prefix in full."""
    actions = np.asarray(actions, dtype=float)
    if actions.ndim != 2 or actions.shape[1] != 14 or not 1 <= len(actions) <= 50 or not np.isfinite(actions).all():
        raise ControlFault('invalid_selected_waypoints')
    output = Path(output)
    output.mkdir()
    seed, state = observation.joints.copy(), observation.state.astype(float).copy()
    blocks = []
    for offset in range(0, len(actions), 5):
        rows = prepare(solver, actions[offset:offset+5], seed, state, output/f'batch-{offset//5+1}', repeats=10)
        blocks.append(rows)
        seed = rows[-1].copy()
        state[:6], state[13:19] = solver.poses(seed)
    joints = np.vstack(blocks)
    if joints.shape != (len(actions), 14):
        raise ControlFault('incomplete_waypoint_plan')
    np.savez_compressed(output/'waypoints.npz', joints=joints, actions=actions, seed=observation.joints)
    return joints


class LoopPlannerIO(PlannerIO):
    """Explicit moving-gripper lane with ownership and acknowledged driver requests."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.commands_lock = threading.RLock()
        self.motion_stopped = False
        self.cancelled = lambda: False

    def require_motion(self):
        self.require_execution()
        if self.motion_stopped or self.cancelled():
            raise ControlFault('motion_cancelled')

    def move(self, joints):
        with self.commands_lock:
            self.require_motion()
            return super().move(joints)

    def select(self, index):
        with self.commands_lock:
            if index == 2:
                self.require_motion()
            else:
                self.motion_stopped = True
            return super().select(index)

    def claim(self):
        self.require_motion()
        g = self.gripper()
        self.boot = g['boot_id']
        if g['reserved']:
            raise ControlFault('gripper_busy_or_restarted')
        return self.grip_request('claim')

    def set_grippers(self, values):
        values = np.asarray(values, dtype=float)
        if values.shape != (2,) or not np.isfinite(values).all() or np.any(values<0) or np.any(values>1):
            raise ControlFault('invalid_gripper_target')
        with self.commands_lock:
            for index, side in enumerate(('left', 'right')):
                self.require_motion()
                g = self.gripper()
                if not self.owns(g) or g['state'] != 'active':
                    raise ControlFault('gripper_session_inactive')
                if abs(g['target'][index]-values[index]) <= 1e-8:
                    continue
                self.seq += 1
                request = dict(action='target', boot_id=self.boot, session_id=self.session,
                               request_id=uuid.uuid4().hex, seq=self.seq, separate_heartbeat=True,
                               side=side, value=float(values[index]), mode='fine',
                               target_generation=g['target_generation'])
                if self.grip_pub.get_subscription_count() < 1:
                    raise ControlFault('gripper_subscriber_missing')
                msg = self.String();msg.data = json.dumps(request, allow_nan=False)
                self.operation_counts['gripper_commands'] += 1
                self.grip_pub.publish(msg);self.publications += 1
                result = self.wait(lambda: (value if (value:=self.gripper()).get('last_request_id') == request['request_id'] else None), 'gripper_target')
                if result.get('result', {}).get('ok') is not True or not self.owns(result):
                    raise ControlFault('gripper_target_rejected_or_ownership_lost')


class LoopRunner(ReplayRunner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gripper_goal = None

    def sample(self, *args, **kwargs):
        try:
            return super().sample(*args, **kwargs)
        except ControlFault as exc:
            # The watchdog can disconnect our source between the cancellation
            # check and feedback read. Keep that race's real cancellation cause.
            if (str(exc) == 'input_or_mode_changed' and self.cancel()
                    and getattr(self.io, 'motion_stopped', False)):
                f = self.io.read()
                if f['input'] == 0 and f['mode'] == 1:
                    raise ControlFault('motion_cancelled') from exc
            raise

    def check_gripper(self, f):
        g = f['gripper']
        if self.gripper_goal is not None and self.claimed:
            if not np.allclose(g['target'], self.gripper_goal, atol=1e-6, rtol=0):
                raise ControlFault('gripper_goal_changed_by_another_writer')

    def acquire(self):
        self.gripper_goal = np.array(self.io.gripper()['applied_target'])
        return super().acquire()

    def execute_segment(self, actions, joints, observed_seed, on_step):
        current = self.io.read()
        check_start(np.array(current['joints']), current['tcp'],
                    observed_seed, np.r_[self.solver.poses(observed_seed)[0], [0]*7,
                                         self.solver.poses(observed_seed)[1], [0]*7])
        previous = np.array(self.io.read()['joints'])
        self.io.allowed_targets = np.vstack((previous, joints))
        self.completed = []
        for index, target in enumerate(joints):
            self.sample(2, previous, previous)
            values = actions[index, [6, 13]]
            self.io.set_grippers(values)
            self.gripper_goal = values.copy()
            self.sample(2, previous, previous)
            self.log('waypoint_submit', step=index+1, joints=target.tolist(), grippers=values.tolist())
            self.io.move(target)
            f = self.wait_arrival(target, 2, start=previous,
                                  tcp_target=np.stack((actions[index,:6], actions[index,7:13])))
            # Gripper goals are acknowledged by the driver; actual position is
            # recorded without a per-waypoint arrival gate or timeout.
            row = dict(step=index+1, measured=f, joint_target=target.tolist(), gripper_target=values.tolist())
            self.completed.append(row)
            self.log('waypoint_reached', **row)
            on_step(index+1)
            previous = target.copy()
        return self.completed


class PlannerLoop:
    """Dependency-injected orchestrator shared by real execution and fault tests."""
    def __init__(self, *, source, policy, planner, runner, config, output, prompt, seed=0,
                 cancel=lambda: False, reason=lambda: 'user_cancelled',
                 pause=lambda: False, resume_state=None, started=None, attempt=1):
        self.source, self.policy, self.planner, self.runner = source, policy, planner, runner
        self.c, self.output, self.prompt, self.seed = config, Path(output), prompt, seed
        self.cancel, self.reason = cancel, reason
        self.pause, self.attempt = pause, attempt
        self.output.mkdir(parents=True, exist_ok=True)
        self.state = dict(id=self.output.name, mode='live', engine='movej', synthetic=False,
                          state='STARTING', completed_cycles=0, cycle=0, history=[],
                          config=vars(config), created_unix=time.time(), hardware_execution_verified=False,
                          timing='retimed_MoveJ_settle_each_waypoint', force_stop_threshold=None)
        if resume_state:
            self.state.update(resume_state)
            for key in ('stop', 'finished_unix', 'reason', 'connection_diagnostics', 'recovery'):
                self.state.pop(key, None)
        self.state['attempt'] = attempt
        self.started = time.monotonic() if started is None else started
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='loop-inference-prepare')
        self.active_future = None
        self.last_id = self.state.get('observation_id')
        self.previous_sources = self.state.get('last_source_times')

    def update(self, state=None, **values):
        if state:
            self.state['state'] = state
        self.state.update(values, updated_unix=time.time())
        save_json(self.output/'status.json', self.state)

    def check_cancel(self):
        if self.cancel():
            raise ControlFault(self.reason())
        if time.monotonic()-self.started >= self.c.max_task_duration_s:
            raise ControlFault('task_time_limit')

    def maintained(self, fn, timeout):
        future = self.pool.submit(fn)
        self.active_future = future
        end = time.monotonic()+timeout
        while not future.done():
            self.check_cancel()
            if time.monotonic() > end:
                raise ControlFault('inference_or_preparation_timeout')
            self.runner.sample(2)
            time.sleep(.02)
        self.check_cancel()
        return future.result()

    def observe(self, barrier):
        from robot_control.closed_loop_observation import ObservationPending
        end = time.monotonic()+self.c.observation_timeout_s
        while True:
            self.check_cancel()
            self.runner.sample(2)
            try:
                o = self.source.read_after(barrier, self.last_id)
                if o.id == self.last_id or min(o.source_times.values()) <= barrier.unix:
                    raise ObservationPending('repeated_or_pre_hold_observation')
                if self.previous_sources and any(o.source_times[k] <= self.previous_sources[k] for k in self.previous_sources):
                    raise ObservationPending('source_not_newer_than_previous_cycle')
                return o
            except ObservationPending as exc:
                if time.monotonic() > end:
                    raise ControlFault('new_observation_timeout:' + str(exc)) from exc
                time.sleep(.03)

    def run(self):
        outcome, reason = 'FAILED', 'unexpected_termination'
        try:
            self.update('ACQUIRE')
            self.check_cancel()
            self.runner.acquire()
            for cycle in range(self.state['completed_cycles']+1, self.c.max_cycles+1):
                self.check_cancel()
                # Fresh camera AND state samples must cross this post-settle barrier.
                barrier = SimpleNamespace(unix=time.time(), monotonic=time.monotonic())
                name = f'{cycle:04d}' if self.attempt == 1 else f'{cycle:04d}-attempt-{self.attempt}'
                folder = self.output/'cycles'/name
                folder.mkdir(parents=True)
                self.update('OBSERVE', cycle=cycle,
                            execution=dict(phase='HOLDING', executed_model_steps=0, session=self.runner.io.session))
                o = self.observe(barrier)
                self.last_id, self.previous_sources = o.id, o.source_times.copy()
                self.state['last_source_times'] = self.previous_sources
                request_id = uuid.uuid4().hex
                np.savez_compressed(folder/'observation.npz', state=o.state, joints=o.joints,
                                    metadata=json.dumps(o.metadata), **o.images)
                record = dict(cycle=cycle, observation_id=o.id, request_id=request_id,
                              source_times=o.source_times, hold_before=vars(barrier))
                self.update('INFER', observation_id=o.id, request_id=request_id)
                started = time.monotonic()
                prediction = self.maintained(lambda: self.policy.predict(
                    {**o.images, 'state':o.state, 'prompt':self.prompt, 'seed':self.seed}, request_id), self.c.rpc_timeout_s+1)
                record['rpc_s'] = time.monotonic()-started
                np.savez_compressed(folder/'prediction.npz', **prediction)
                actions, correction = adapt(prediction['actions'], self.c.max_gripper_correction, execution_horizon=self.c.execution_horizon)
                record['max_gripper_correction'] = correction
                record['execution_horizon'] = self.c.execution_horizon
                self.update('PREPARE', rpc_s=record['rpc_s'])
                started = time.monotonic()
                joints = self.maintained(lambda: self.planner(actions, o, folder/'plan'), self.c.prepare_timeout_s)
                record['prepare_s'] = time.monotonic()-started
                joints = np.asarray(joints)
                if joints.shape != (self.c.execution_horizon, 14) or not np.isfinite(joints).all():
                    raise ControlFault('prepared_waypoint_count_or_values_invalid')
                f = self.runner.sample(2)
                check_start(np.array(f['joints']), f['tcp'], o.joints, o.state)
                self.check_cancel()
                self.update('EXECUTE', prepare_s=record['prepare_s'],
                            prediction_age_s=time.monotonic()-o.observed_at)
                def on_step(n):
                    self.update(execution=dict(phase='EXECUTE', executed_model_steps=n, session=self.runner.io.session),
                                prediction_age_s=time.monotonic()-o.observed_at)
                result = self.runner.execute_segment(actions, joints, o.joints, on_step)
                self.update('SETTLING')
                record.update(executed_steps=len(result), hold_confirmed=True, completed_unix=time.time())
                save_json(folder/'reached.json', result)
                save_json(folder/'cycle.json', record)
                save_json(folder/'feedback.json', self.runner.trace)
                self.runner.trace.clear()
                if len(result) != self.c.execution_horizon:
                    raise ControlFault('incomplete_selected_steps')
                self.state['history'].append(record)
                self.update(completed_cycles=cycle, hardware_execution_verified=True)
            outcome, reason = 'COMPLETED_LIMIT', 'configured_cycle_limit'
        except Exception as exc:
            reason = str(exc)
            interrupted = self.cancel()
            if interrupted and reason in ('operator_cancelled', 'motion_cancelled',
                                           'browser_heartbeat_timeout', 'user_cancelled',
                                           'parent_disconnected', 'task_time_limit', 'service_stopping'):
                reason = self.reason()
            outcome = ('TIMEOUT' if reason == 'task_time_limit' else
                       'PAUSING' if reason == 'browser_heartbeat_timeout' and self.pause() else
                       'CANCELLED' if interrupted else 'FAILED')
        finally:
            self.update('STOPPING')
            try:
                stop = self.runner.stop()
                confirmed = (stop.get('no_ownership_acquired') or
                             (stop.get('input_none') and stop.get('stationary_confirmed') and
                              (stop.get('gripper_released') or stop.get('claim_not_acquired'))))
                stop['confirmed'] = bool(confirmed)
                if not confirmed:
                    outcome = 'STOP_UNCONFIRMED'
                elif outcome == 'PAUSING':
                    outcome = 'PAUSED'
            except Exception as exc:
                outcome, stop = 'STOP_UNCONFIRMED', dict(confirmed=False, error=str(exc))
            self.update(outcome, reason=reason, stop=stop,
                        finished_unix=None if outcome == 'PAUSED' else time.time())
            save_json(self.output/f'final_feedback-attempt-{self.attempt}.json', self.runner.trace)
            save_json(self.output/'final_feedback.json', self.runner.trace)
            self.pool.shutdown(wait=False, cancel_futures=True)
        return self.state
