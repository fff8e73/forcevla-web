"""Explicit replay of the ten saved TCP targets through the installed MoveJ planner.

This is a newly planned, stop-at-each-waypoint replay, NOT a fresh model prediction
or 30 Hz playback. Original observations and their timestamps remain unchanged.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_control.command_bridge import ControlFault
from robot_control.common import sha256, save_json
from robot_control.offline_ik import JOINTS, pose_delta

SOURCE = Path(__file__).resolve().parents[1] / 'data/integration-runs/20261014-211413-current-first10'


def load_targets(folder=SOURCE):
    folder = Path(folder)
    meta = json.loads((folder / 'metadata.json').read_text())
    for name in ('first10.npz', 'observation.npz', 'prediction_all50.npz'):
        if sha256(folder / name) != meta['artifact_sha256'][name]:
            raise ControlFault('source_hash_mismatch:' + name)
    with np.load(folder / 'first10.npz', allow_pickle=False) as d:
        raw = d['raw_actions'].astype(float)
    with np.load(folder / 'prediction_all50.npz', allow_pickle=False) as d:
        if not np.array_equal(raw, d['actions'][0, :10]):
            raise ControlFault('source_prefix_mismatch')
    with np.load(folder / 'observation.npz', allow_pickle=False) as d:
        state = d['state'].astype(float)
        observation = json.loads(str(d['metadata']))
    if raw.shape != (10, 14) or not np.isfinite(raw).all():
        raise ControlFault('invalid_ten_actions')
    if observation['joints']['names'] != JOINTS:
        raise ControlFault('source_joint_order')
    bounded = raw.copy()
    bounded[:, [6, 13]] = np.clip(raw[:, [6, 13]], 0, 1)
    if np.max(np.abs(raw - bounded)) > .05:
        raise ControlFault('gripper_correction_too_large')
    # This replay handles these particular ten identical zero gripper goals.
    if not np.array_equal(bounded[:, [6, 13]], np.zeros((10, 2))):
        raise ControlFault('this_replay_requires_fixed_zero_gripper_targets')
    return raw, bounded, state, np.array(observation['joints']['positions']), meta


def check_start(q, tcp, source_q, source_state):
    if np.max(np.abs(q - source_q)) > .002:
        raise ControlFault('current_joints_differ_from_recorded_start')
    errors = [pose_delta(a, b) for a, b in zip(
        tcp, np.stack((source_state[:6], source_state[13:19])))]
    if any(p > .003 or r > .02 for p, r in errors):
        raise ControlFault('current_tcp_differs_from_recorded_start')
    return errors


def prepare(solver, actions, seed, state, output, *, repeats=5):
    """Use the installed QP core for all endpoints; never call the robot's IK topic."""
    output = Path(output)
    output.mkdir(exist_ok=False)
    # Five identical targets give the native solver more settling time per goal.
    # These are offline integration ticks, not real execution timestamps.
    actions = np.asarray(actions, dtype=float)
    if (type(repeats) is not int or repeats <= 0 or actions.ndim != 2
            or actions.shape[1] != 14 or not 1 <= len(actions)*repeats <= 50
            or not np.isfinite(actions).all()):
        raise ValueError('Offline waypoint batch requires 1..50 finite settling targets')
    expanded = np.repeat(actions, repeats, axis=0)
    # Keep the fixed 50-row solver input contract, but solve only real targets.
    # Padding must never become an extra executed waypoint in a partial batch.
    padded = np.vstack((expanded, np.repeat(expanded[-1:], 50-len(expanded), axis=0)))
    result = solver.rollout(padded, seed, state, output, prefix_steps=len(expanded))
    save_json(output / 'qp_checks.json', result)
    if result['status'] != 'passed_scoped_qp' or result['reasons']:
        raise ControlFault('offline_qp_rejected:' + str(result['reasons']))
    with np.load(output / 'qp_trajectory.npz', allow_pickle=False) as d:
        q = d['joint_positions']
        ids = d['target_indices']
        waypoints = np.array([q[np.flatnonzero(ids == repeats*(i+1)-1)[-1]] for i in range(len(actions))])
    residuals = []
    min_distance = float('inf')
    previous = np.array(seed)
    for i, row in enumerate(waypoints):
        errors = [pose_delta(a, b) for a, b in zip(
            solver.poses(row), np.stack((actions[i, :6], actions[i, 7:13])))]
        if any(p > .001 or r > .01 for p, r in errors):
            raise ControlFault('waypoint_ik_residual:' + str(i + 1))
        residuals.append(errors)
        # Diagnostic straight joint paths. Native planner rehearsal additionally
        # checks its actual per-joint trapezoidal path, which can differ.
        for alpha in np.linspace(0, 1, 101):
            distances = solver.collisions(previous + alpha * (row - previous))
            near = min(x['distance_m'] for x in distances)
            min_distance = min(min_distance, near)
            if near < solver.parameters['collision_minimum_distance']:
                raise ControlFault('configured_collision_margin:' + str(i + 1))
        previous = row
    np.savez_compressed(output / 'waypoints.npz', joints=waypoints, seed=seed, actions=actions)
    save_json(output / 'waypoints.json', dict(
        targets=actions.tolist(), joints=waypoints.tolist(), residuals=residuals,
        min_configured_distance_m=min_distance, environment_collision_checked=False,
        execution_timing='installed MoveJ trapezoidal planner; settle at each waypoint',
        original_model_period_s=1/30, hardware_execution_verified=False))
    return waypoints


def pose6(message):
    p, q = message.pose.position, message.pose.orientation
    return np.r_[[p.x, p.y, p.z], Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz')]


class ReplayRunner:
    """Small, sequential transaction; IO supplies fresh samples and bounded RPCs."""
    def __init__(self, io, solver, actions, waypoints, seed, *, clock=time.monotonic,
                 sleep=time.sleep, cancel=lambda: False, log=lambda event, **kw: None):
        self.io, self.solver = io, solver
        self.actions, self.waypoints, self.seed = actions, waypoints, seed
        self.clock, self.sleep, self.cancel, self.log = clock, sleep, cancel, log
        self.claimed = self.selected = False
        self.completed = []
        self.trace = []
        self.baseline_wrench = None
        self.last_heartbeat = float('-inf')
        self.last_poll = clock()
        self.stopped = False

    def check_gripper(self, feedback):
        if not np.allclose(feedback['gripper']['applied_target'], [0, 0], atol=1e-8, rtol=0):
            raise ControlFault('fixed_gripper_target_changed')

    def sample(self, expected_input, start=None, target=None):
        if self.cancel():
            raise ControlFault('operator_cancelled')
        f = self.io.read()
        if f['input'] != expected_input or f['mode'] != 1:
            raise ControlFault('input_or_mode_changed')
        if self.claimed and not self.io.owns(f['gripper']):
            raise ControlFault('gripper_ownership_lost')
        if self.claimed and f['gripper']['state'] != 'active':
            raise ControlFault('gripper_session_not_active')
        self.check_gripper(f)
        q, velocity = np.array(f['joints']), np.array(f['velocities'])
        if np.any(q < self.solver.bounds[:, 0]) or np.any(q > self.solver.bounds[:, 1]):
            raise ControlFault('joint_position_limit')
        # Feedback tolerance is explicitly separate from configured commanded speed.
        if np.any(np.abs(velocity) > self.io.velocity + .05):
            raise ControlFault('measured_speed_exceeds_planner_budget_plus_0.05')
        if start is not None:
            lo, hi = np.minimum(start, target) - .01, np.maximum(start, target) + .01
            if np.any(q < lo) or np.any(q > hi):
                raise ControlFault('measured_joints_outside_current_segment')
            if expected_input == 2:
                command = np.array(f['joint_target'])
                if np.any(command < lo) or np.any(command > hi):
                    raise ControlFault('unexpected_mux_target')
                if np.max(np.abs(q-command)) > .02:
                    raise ControlFault('joint_tracking_error')
        if min(x['distance_m'] for x in self.solver.collisions(q)) < .005:
            raise ControlFault('configured_robot_collision_margin')
        if self.baseline_wrench is not None:
            # Uncalibrated raw wrench is diagnostic only. No invented force gate.
            f['raw_wrench_delta'] = (np.array(f['wrench']) - self.baseline_wrench).tolist()
        if self.claimed and self.clock() - self.last_heartbeat >= .08:
            self.io.heartbeat()
            self.last_heartbeat = self.clock()
        self.last_poll = self.clock()
        self.trace.append(f)
        return f

    def wait_arrival(self, target, expected_input, *, start, min_duration=.3, timeout=8.,
                     tcp_target=None, planner_only=False):
        begin = self.clock()
        settled_at = None
        last_sample = None
        while self.clock() - begin < timeout:
            f = self.sample(expected_input, start, target)
            value = f['planner_target'] if planner_only else f['joints']
            error = float('inf') if value is None else float(np.max(np.abs(np.array(value) - target)))
            stopped = np.max(np.abs(f['velocities'])) <= .02
            in_place = error <= (.00001 if planner_only else .003)
            if tcp_target is not None:
                errors = [pose_delta(a, b) for a, b in zip(f['tcp'], tcp_target)]
                in_place &= all(p <= .002 and r <= .02 for p, r in errors)
            if not in_place or not stopped:
                settled_at = None
            elif last_sample != f['sample_time']:
                if settled_at is None:
                    settled_at = self.clock()
                if self.clock() - settled_at >= min_duration:
                    return f
            last_sample = f['sample_time']
            self.sleep(.02)
        raise ControlFault('waypoint_arrival_timeout')

    def acquire(self):
        f = self.sample(0)
        if np.max(np.abs(np.array(f['joints']) - self.seed)) > .002:
            raise ControlFault('seed_changed_since_preparation')
        if np.max(np.abs(f['velocities'])) > .02 or f['gripper']['reserved']:
            raise ControlFault('not_stationary_or_gripper_busy')
        self.baseline_wrench = np.array(f['wrench'])
        # Set intent before RPC: a timeout may still mean the claim was applied.
        self.claimed = True
        self.io.claim()
        self.log('gripper_claimed', target=f['gripper']['applied_target'])
        self.sample(0)
        # Erase any old planner goal while its output is disconnected.
        self.io.move(self.seed)
        self.wait_arrival(self.seed, 0, start=self.seed, min_duration=.1, planner_only=True)
        self.selected = True  # record ambiguous selector outcome as ownership intent
        self.io.select(2)
        self.log('planner_selected')
        # The mux starts forwarding new planner samples only after selection.
        # Reissue the same validated stationary target to initialize this lane.
        self.io.move(self.seed)
        self.sleep(.05)
        self.wait_arrival(self.seed, 2, start=self.seed, min_duration=3.3, timeout=6.)

    def run(self):
        self.acquire()
        previous = self.seed.copy()
        for index, target in enumerate(self.waypoints):
            self.sample(2, previous, previous)
            self.log('waypoint_submit', step=index + 1, joints=target.tolist(),
                     tcp=np.stack((self.actions[index, :6], self.actions[index, 7:13])).tolist())
            started = self.clock()
            self.io.move(target)
            f = self.wait_arrival(target, 2, start=previous,
                                  tcp_target=np.stack((self.actions[index, :6], self.actions[index, 7:13])))
            result = dict(step=index + 1, duration_s=self.clock()-started, measured=f,
                          target_joint_error_rad=float(np.max(np.abs(np.array(f['joints'])-target))))
            self.completed.append(result)
            self.log('waypoint_reached', **result)
            previous = target.copy()
        return self.completed

    def stop(self):
        """Disconnect the planner first, then independently hold/release grippers."""
        if self.stopped:
            return self.stop_report
        self.stopped = True
        report = dict(input_none=False, gripper_held=False, stationary_confirmed=False,
                      gripper_released=False, errors=[])
        self.stop_report = report
        if not self.claimed and not self.selected:
            report['no_ownership_acquired'] = True
            return report
        if self.selected:
            try:
                # Only disconnect our source; never seize another active source.
                current = self.io.input()
                if current == 2:
                    self.io.select(0)
                elif current != 0:
                    raise ControlFault('control_source_changed_do_not_overwrite')
                report['input_none'] = self.io.input() == 0
            except Exception as exc:
                report['errors'].append('input_stop:' + str(exc))
        else:
            report['input_none'] = self.io.input() == 0
        if self.claimed:
            try:
                if self.io.owns(self.io.gripper()):
                    self.io.hold()
                    report['gripper_held'] = True
                elif not self.io.gripper().get('reserved'):
                    report['claim_not_acquired'] = True
            except Exception as exc:
                report['errors'].append('gripper_hold:' + str(exc))
        begin, settled, last = self.clock(), None, None
        while self.clock() - begin < 3:
            try:
                f = self.io.read()
                if f['input'] != 0 or f['mode'] != 1 or np.max(np.abs(f['velocities'])) > .02:
                    settled = None
                elif f['sample_time'] != last:
                    if settled is None:
                        settled = self.clock()
                    if self.clock() - settled >= .3:
                        report['stationary_confirmed'] = True
                        report['final_feedback'] = f
                        break
                last = f['sample_time']
            except Exception as exc:
                settled = None
                report['last_feedback_error'] = str(exc)
            self.sleep(.02)
        if report['input_none'] and report['stationary_confirmed'] and report['gripper_held']:
            try:
                self.io.release()
                report['gripper_released'] = not self.io.gripper()['reserved']
            except Exception as exc:
                report['errors'].append('gripper_release:' + str(exc))
        self.log('stop_result', **report)
        return report
