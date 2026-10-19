#!/usr/bin/env python3
"""Replay the referenced ten targets. Default: prepare/check only; --execute moves hardware."""
import argparse
import fcntl
import json
from pathlib import Path
import signal
import threading
import time
import uuid

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import CONTROL_ROOT, ROOT, save_json, sha256
from robot_control.first10_replay import SOURCE, ReplayRunner, check_start, load_targets, pose6, prepare
from robot_control.native_qp import NativeQP
from robot_control.ros_execution_transport import RosExecutionTransport
from robot_control.diagnostics.measured_hold_probe import read_status


class PlannerIO(RosExecutionTransport):
    """MoveJ service client; never publishes joint commands or shared IK requests."""
    def __init__(self, *, execute=False, domain=24, namespace='/tj'):
        self.execute = execute
        super().__init__(domain=domain, namespace=namespace, allow_missing_targets=True,
                         timeout_s=.8)
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from marvin_msgs.msg import Jointcmd
        from marvin_msgs.srv import MoveJ
        from rcl_interfaces.srv import GetParameters
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        self.MoveJ, self.GetParameters = MoveJ, GetParameters
        self.move_client = self.node.create_client(MoveJ, namespace + '/control/movej')
        self.params_client = self.node.create_client(GetParameters, namespace + '/planner_joint_node/get_parameters')
        for side, letter in [('left', 'A'), ('right', 'B')]:
            for kind, topic, key in [
                (PoseStamped, '/info/eef_' + side, 'pose_' + side),
                (WrenchStamped, '/info/wrench_' + side, 'wrench_' + side),
                (Jointcmd, '/control/joint_cmd_plan_' + letter, 'plan_' + side),
            ]:
                self.subs.append(self.node.create_subscription(kind, namespace + topic, self.callback(key), sensor))
        self.velocity = self.acceleration = None
        self.session = str(uuid.uuid4())
        self.boot = None
        self.seq = self.beat_seq = 0
        self.allowed_targets = None
        self.operation_counts['movej'] = 0

    def rpc(self, client, request):
        if not client.service_is_ready():
            raise ControlFault('service_not_ready:' + client.srv_name)
        future = client.call_async(request)
        return self.wait(lambda: future.result() if future.done() else None, client.srv_name)

    def parameters(self):
        request = self.GetParameters.Request()
        request.names = ['home_vel_limit', 'home_acc_limit']
        result = self.rpc(self.params_client, request)
        self.velocity, self.acceleration = [np.array(v.double_array_value) for v in result.values]
        for value in (self.velocity, self.acceleration):
            if value.shape != (14,) or not np.isfinite(value).all() or np.any(value <= 0):
                raise ControlFault('invalid_planner_limits')
        # Pin the actual configuration reviewed for this replay; do not change it.
        if not np.allclose(self.velocity, .2) or not np.allclose(self.acceleration, .5):
            raise ControlFault('planner_limits_changed_since_review')
        return dict(velocity_rad_s=self.velocity.tolist(), acceleration_rad_s2=self.acceleration.tolist(),
                    source=self.namespace + '/planner_joint_node:get_parameters', hardware_ratings=False)

    def require_execution(self):
        if not self.execute:
            raise ControlFault('check_only_no_writes')

    def move(self, joints):
        self.require_execution()
        q = np.asarray(joints)
        if self.allowed_targets is None or not any(np.array_equal(q, row) for row in self.allowed_targets):
            raise ControlFault('unapproved_waypoint')
        request = self.MoveJ.Request()
        request.joint_values = q.tolist()
        self.operation_counts['movej'] += 1
        if not self.rpc(self.move_client, request).success:
            raise ControlFault('movej_rejected')

    def select(self, index):
        self.require_execution()
        if index == 0:
            return super().select_input(0)
        if index != 2:
            raise ControlFault('only_none_or_planner')
        request = self.Int.Request()
        request.data = 2
        started = time.monotonic()
        self.operation_counts['input_requests'] += 1
        if not self.rpc(self.selector, request).success:
            raise ControlFault('planner_selection_rejected')
        self.wait(lambda: True if self.sample('input')[0].data == 2 and self.sample('input')[1] >= started else None,
                  'planner_selection_readback')

    def input(self):
        with self.lock:
            return int(self.sample('input')[0].data)

    def gripper(self):
        return self.gripper_status()

    def owns(self, status):
        return status.get('reserved') is True and status.get('boot_id') == self.boot and status.get('session_id') == self.session

    def grip_request(self, action):
        self.require_execution()
        self.seq += 1
        request = dict(action=action, boot_id=self.boot, session_id=self.session,
                       request_id=uuid.uuid4().hex, seq=self.seq, separate_heartbeat=True)
        response = super().gripper_command(request)
        if response.get('result', {}).get('ok') is not True:
            raise ControlFault('gripper_' + action + '_rejected:' + str(response.get('result')))
        if action != 'release' and not self.owns(response):
            raise ControlFault('gripper_ownership_not_confirmed')
        return response

    def claim(self):
        self.require_execution()
        g = self.gripper()
        self.boot = g['boot_id']
        if g['reserved']:
            raise ControlFault('gripper_became_reserved')
        return self.grip_request('claim')

    def heartbeat(self):
        self.require_execution()
        self.beat_seq += 1
        return super().heartbeat(dict(boot_id=self.boot, session_id=self.session, beat_seq=self.beat_seq))

    def hold(self):
        result = self.grip_request('hold')
        if result.get('state') != 'held':
            raise ControlFault('gripper_hold_not_confirmed')

    def release(self):
        result = self.grip_request('release')
        if result.get('reserved') or result.get('session_id') is not None:
            raise ControlFault('gripper_release_not_confirmed')

    def read(self):
        with self.lock:
            f = super().feedback()
            now = time.monotonic()
            if not f.healthy or not 0 <= now-f.received_at <= .3 or not 0 <= f.source_age_s <= .3:
                raise ControlFault('joint_feedback_stale_or_unhealthy')
            if f.unexpected_publishers:
                raise ControlFault('unexpected_user_publisher')
            for a in (f.joints, f.velocities):
                if a.shape != (14,) or not np.isfinite(a).all():
                    raise ControlFault('invalid_joint_feedback')
            if f.selected_input == 2:
                if not np.isfinite(f.joint_target).all():
                    raise ControlFault('mux_target_missing')
                # Native MoveJ stops publishing after reaching its final target.
                # Retain the original command timestamp; use independent, fresh
                # joint measurements to confirm the subsequent stationary hold.
                if now-f.target_received_at > .3 and np.max(np.abs(f.velocities)) > .02:
                    raise ControlFault('moving_with_stale_mux_target')
            g = f.gripper
            if not g.get('healthy') or np.any(np.array(g['motor_feedback_age']) > .3):
                raise ControlFault('gripper_feedback_stale_or_unhealthy')
            for key in ('actual', 'applied_target', 'motor_feedback_age', 'motor_velocities'):
                value = np.array(g[key])
                if value.shape != (2,) or not np.isfinite(value).all():
                    raise ControlFault('invalid_gripper_feedback:' + key)
            if np.any(np.array(g['motor_feedback_age']) < 0):
                raise ControlFault('invalid_motor_feedback_age')
            if not np.allclose(g['calibration'], [[0, 1.6], [0, 1.6]]):
                raise ControlFault('gripper_calibration_changed')
            tcp, wrench, planner = [], [], []
            for side, letter in [('left', 'L'), ('right', 'R')]:
                pose, at = self.sample('pose_' + side)
                force, force_at = self.sample('wrench_' + side)
                for msg, received in [(pose, at), (force, force_at)]:
                    age = time.time() - msg.header.stamp.sec - msg.header.stamp.nanosec/1e9
                    if not 0 <= now-received <= .3 or not 0 <= age <= .3:
                        raise ControlFault('tcp_or_wrench_stale')
                if pose.header.frame_id != 'base_link' or force.header.frame_id != 'flange_' + letter:
                    raise ControlFault('tcp_or_wrench_frame_changed')
                tcp.append(pose6(pose).tolist())
                w = force.wrench
                wrench.append([w.force.x, w.force.y, w.force.z, w.torque.x, w.torque.y, w.torque.z])
                if 'plan_' + side in self.samples:
                    p, received = self.sample('plan_' + side)
                    planner.extend(list(p.positions) if now-received <= .3 else [float('nan')]*7)
                else:
                    planner.extend([float('nan')]*7)
            if not np.isfinite(tcp).all() or not np.isfinite(wrench).all():
                raise ControlFault('invalid_tcp_or_wrench')
            return dict(sample_time=f.received_at, input=f.selected_input, mode=f.mode,
                        joints=f.joints.tolist(), velocities=f.velocities.tolist(),
                        joint_target=f.joint_target.tolist() if np.isfinite(f.joint_target).all() else None,
                        joint_target_age_s=now-f.target_received_at if f.target_received_at else None,
                        planner_target=planner if np.isfinite(planner).all() else None,
                        tcp=tcp, wrench=wrench, gripper=g)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true', help='Execute this explicitly requested saved-target replay')
    p.add_argument('--output', type=Path)
    p.add_argument('--resume-stopped-run', type=Path, help='Explicit recovery from a confirmed stopped run with no completed waypoint')
    args = p.parse_args()
    output = args.output or ROOT / 'data/integration-runs' / (time.strftime('%Y%m%d-%H%M%S') + '-first10-replay')
    output.mkdir(parents=True, exist_ok=False)
    lock = open('/tmp/forcevla-first10-replay.lock', 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cancel = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: cancel.set())
    report = dict(status='failed', execute_requested=args.execute, source=str(SOURCE),
                  semantics='explicit saved-target waypoint replay, independently retimed; no model timestamp renewal',
                  hardware_motion_verified=False, model_task_success_evaluated=False,
                  source_code_sha256={name: sha256(CONTROL_ROOT/name) for name in ('run_first10.py', 'first10_replay.py')})
    io = runner = None
    watcher_stop = threading.Event()
    watcher = None
    log_lock = threading.Lock()

    def log(event, **fields):
        row = dict(time=time.time(), event=event, **fields)
        with log_lock, (output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
        if event in ('waypoint_submit', 'waypoint_reached'):
            print(json.dumps(dict(event=event, step=fields['step']), ensure_ascii=False), flush=True)

    try:
        raw, actions, saved_state, saved_q, meta = load_targets()
        report['source_request_id'] = meta['request_id']
        report['source_captured_at'] = meta['captured_at']
        report['source_age_s_at_start'] = time.time() - __import__('datetime').datetime.fromisoformat(meta['captured_at']).timestamp()
        report['wrench_policy'] = 'raw diagnostic log only; no uncalibrated force/torque stop threshold'
        report['gripper_policy'] = 'explicit saturation to [0,1]; all ten equal current target [0,0]'
        report['max_gripper_correction'] = float(np.max(np.abs(raw-actions)))
        report['http_before'] = read_status()
        limits = json.loads((ROOT/'config/planning.json').read_text())
        solver = NativeQP(limits['expected_qp_parameters'], limits)
        for path, expected in limits['solver_file_sha256'].items():
            if sha256(path) != expected:
                raise ControlFault('model_or_solver_hash_changed:' + path)
        io = PlannerIO(execute=args.execute)
        end = time.monotonic() + 8
        while True:
            try:
                f = io.read()
                if io.move_client.service_is_ready() and io.params_client.service_is_ready() and io.selector.service_is_ready():
                    break
            except ControlFault:
                pass
            if time.monotonic() > end:
                raise ControlFault('feedback_or_services_unavailable')
            time.sleep(.05)
        report['planner_limits'] = io.parameters()
        seed = np.array(f['joints'])
        reference_q, reference_state = saved_q, saved_state
        if args.resume_stopped_run:
            prior_path = args.resume_stopped_run / 'report.json'
            prior = json.loads(prior_path.read_text())
            stop = prior.get('stop', {})
            if (prior.get('source_request_id') != meta['request_id']
                    or prior.get('completed_steps') != 0
                    or prior.get('execute_requested') is not True
                    or not all(stop.get(k) is True for k in ('input_none', 'stationary_confirmed', 'gripper_released'))):
                raise ControlFault('resume_requires_matching_source_and_confirmed_stop_before_first_waypoint')
            last = stop['final_feedback']
            reference_q = np.array(last['joints'])
            reference_state = saved_state.copy()
            reference_state[:6], reference_state[13:19] = last['tcp']
            report['explicit_recovery'] = dict(prior_report=str(prior_path), prior_sha256=sha256(prior_path),
                                               previous_error=prior.get('error'), first_unfinished_step=1,
                                               action_targets_unchanged=True)
        report['start_errors'] = check_start(seed, f['tcp'], reference_q, reference_state)
        report['before'] = f
        state = saved_state.copy()
        state[:6], state[13:19] = f['tcp']
        waypoints = prepare(solver, actions, seed, state, output/'prepared')
        io.allowed_targets = np.vstack((seed, waypoints))
        runner = ReplayRunner(io, solver, actions, waypoints, seed, cancel=cancel.is_set, log=log)
        # Explicitly check the start again after all offline work.
        after = runner.sample(0)
        check_start(np.array(after['joints']), after['tcp'], seed, state)
        save_json(output/'preflight.json', dict(before=f, after=after))
        if not args.execute:
            report['status'] = 'prepared_checked_no_hardware_commands'
        else:
            report['http_before_acquire'] = read_status()
            runner.last_poll = time.monotonic()

            def watchdog():
                while not watcher_stop.wait(.05):
                    if runner.selected and not runner.stopped and time.monotonic()-runner.last_poll > 1.0:
                        cancel.set()
                        try:
                            if io.input() == 2:
                                io.select(0)
                            log('watchdog_disconnected_planner')
                        except Exception as exc:
                            log('watchdog_stop_unconfirmed', error=str(exc))
                        return

            watcher = threading.Thread(target=watchdog, name='first10-watchdog', daemon=True)
            watcher.start()
            runner.run()
            stop = runner.stop()
            if not (stop['stationary_confirmed'] and stop['input_none'] and stop['gripper_released']):
                raise ControlFault('final_stop_or_release_unconfirmed')
            report.update(status='completed_ten_waypoints_and_stopped', hardware_motion_verified=True)
    except Exception as exc:
        report['error'] = repr(exc)
    finally:
        if runner:
            try:
                report['stop'] = runner.stop()
            except Exception as exc:
                report['stop_error'] = repr(exc)
            report['completed_steps'] = len(runner.completed)
            save_json(output/'completed_steps.json', runner.completed)
            save_json(output/'feedback.json', runner.trace)
        watcher_stop.set()
        if watcher:
            watcher.join(timeout=2)
        if io:
            report['operation_counts'] = io.operation_counts
            io.close()
        try:
            report['http_after'] = read_status()
        except Exception as exc:
            report['http_after_error'] = str(exc)
        save_json(output/'report.json', report)
        lock.close()
    print(json.dumps(dict(status=report['status'], completed_steps=report.get('completed_steps', 0),
                          error=report.get('error'), report=str(output/'report.json')), ensure_ascii=False), flush=True)
    return 0 if report['status'] in ('prepared_checked_no_hardware_commands', 'completed_ten_waypoints_and_stopped') else 2


if __name__ == '__main__':
    raise SystemExit(main())
