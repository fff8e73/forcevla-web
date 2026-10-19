#!/usr/bin/env python3
"""Web child process for current-observation, configurable-waypoint MoveJ loops."""
import argparse
import fcntl
import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot_control.planner_supervision import PlannerSupervision
from robot_control.planner_recovery import require_unchanged_hold, run_recovering
from robot_control.closed_loop_observation import FileObservationSource
from robot_control.command_bridge import ControlFault
from robot_control.common import CONTROL_ROOT, ROOT, save_json, sha256
from robot_control.diagnostics.measured_hold_probe import read_status
from robot_control.native_qp import NativeQP
from robot_control.offline_ik import OfflineIK
from robot_control.planner_loop import LoopPlannerIO, LoopRunner, PlannerLoop, prepare_waypoints, profile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['live'], required=True)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--runtime', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--prompt', required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--parent-control', action='store_true')
    args = p.parse_args()
    c = profile(json.loads(args.config.read_text()))
    runtime = json.loads(args.runtime.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output/'status.json').exists():
        raise SystemExit('Existing runs cannot resume')
    supervision = PlannerSupervision(c, track_parent=args.parent_control)
    started = supervision.started
    watchdog_stop = threading.Event()
    watcher = None
    operation_totals = {}
    io = runner = loop = policy = None
    lock = (ROOT / 'data/integration-runs/planner-control.lock').open('a')
    # Share the original one-shot executor lock as well.
    replay_lock = open('/tmp/forcevla-first10-replay.lock', 'a')
    def cancel(reason):
        supervision.cancel(reason)
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: cancel('service_stopping'))
    def cancellation_requested():
        return supervision.interrupted()
    if args.parent_control:
        def read_parent():
            try:
                while line := sys.stdin.readline(8192):
                    data = json.loads(line)
                    if data.get('command') == 'stop':
                        cancel('user_cancelled')
                    elif data.get('command') == 'heartbeat':
                        supervision.heartbeat(data.get('seen_monotonic'), data.get('connection_health'))
                cancel('parent_disconnected')
            except Exception:
                cancel('invalid_parent_message')
        threading.Thread(target=read_parent, daemon=True, name='planner-loop-parent').start()
    log_lock = threading.Lock()
    def log(event, **fields):
        with log_lock, (args.output/'events.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(time=time.time(), event=event, **fields), allow_nan=False)+'\n')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(replay_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save_json(args.output/'status.json', dict(id=args.output.name, state='PRECHECK', mode='live', engine='movej', completed_cycles=0, config=vars(c)))
        save_json(args.output/'http_before.json', read_status())
        limits = json.loads((ROOT/'config/planning.json').read_text())
        for path, expected in limits['solver_file_sha256'].items():
            if sha256(path) != expected:
                raise ControlFault('model_or_solver_asset_changed:' + path)
        solver = NativeQP(limits['expected_qp_parameters'], limits)
        geometry = OfflineIK(limits['expected_qp_parameters'], limits)
        sys.path.insert(0, runtime['web_root'])
        from forcevla_web.policy import RemotePolicy
        policy = RemotePolicy(dict(policy_repo=runtime['policy_repo'], model_address=runtime['model_address'], rpc_timeout_s=c.rpc_timeout_s))
        if not policy.healthy():
            raise ControlFault('model_service_unavailable')
        def make_loop(prior=None, held=None, attempt=1):
            nonlocal io, runner, loop, watcher, watchdog_stop
            watchdog_stop.set()
            if watcher:
                watcher.join(timeout=2)
                if watcher.is_alive():
                    raise ControlFault('watchdog_did_not_exit')
            if io:
                for key, value in io.operation_counts.items():
                    operation_totals[key] = operation_totals.get(key, 0)+value
                io.close()
                io = None
            runner = None
            io = LoopPlannerIO(execute=True)
            io.cancelled = cancellation_requested
            deadline = time.monotonic()+8
            while True:
                if cancellation_requested() and supervision.terminal_reason:
                    raise ControlFault(supervision.reason())
                try:
                    f = io.read()
                    if io.selector.service_is_ready() and io.params_client.service_is_ready() and io.move_client.service_is_ready():
                        break
                except ControlFault:
                    pass
                if time.monotonic()>deadline:
                    raise ControlFault('robot_feedback_or_planner_service_unavailable')
                time.sleep(.03)
            if held is not None:
                require_unchanged_hold(f, held)
            save_json(args.output/f'planner_limits-attempt-{attempt}.json', io.parameters())
            seed = np.array(f['joints'])
            io.allowed_targets = seed[None].copy()
            runner = LoopRunner(io, geometry, None, None, seed, cancel=cancellation_requested, log=log)
            watchdog_stop = threading.Event()
            def watchdog():
                while not watchdog_stop.wait(.05):
                    aborted = cancellation_requested()
                    if runner.selected and not runner.stopped and (aborted or time.monotonic()-runner.last_poll > 2):
                        if not aborted:
                            cancel('feedback_supervision_stalled')
                        try:
                            if io.input() == 2:
                                io.select(0)
                            log('watchdog_stop_requested', reason=supervision.reason())
                        except Exception as exc:
                            log('watchdog_stop_unconfirmed', error=str(exc))
                        return
            watcher = threading.Thread(target=watchdog, daemon=True, name='planner-loop-stop')
            watcher.start()
            loop = PlannerLoop(source=FileObservationSource(runtime['ipc_path'], c), policy=policy,
                               planner=lambda a,o,p:prepare_waypoints(solver,a,o,p), runner=runner, config=c,
                               output=args.output, prompt=args.prompt, seed=args.seed,
                               cancel=cancellation_requested, reason=supervision.reason,
                               pause=lambda:bool(supervision.pause_reason and not supervision.terminal_reason),
                               resume_state=prior, started=started, attempt=attempt)
            return loop
        loop = make_loop()
        save_json(args.output/'identity.json', dict(
            source_sha256={name:sha256(CONTROL_ROOT/name) for name in ('planner_loop.py','run_planner_loop.py','planner_supervision.py','planner_recovery.py','first10_replay.py','run_first10.py')},
            solver=solver.manifest, model_address=runtime['model_address']))
        result = run_recovering(loop, supervision, make_loop, log=log)
        result['connection_diagnostics'] = supervision.diagnostics or dict(
            captured_unix=time.time(), connection_health=supervision.health)
        save_json(args.output/'connection_diagnostics.json', result['connection_diagnostics'])
        save_json(args.output/'status.json', result)
        print(json.dumps({k:result.get(k) for k in ('id','state','completed_cycles','reason')}, ensure_ascii=False), flush=True)
        return 0 if result['state'] in ('COMPLETED_LIMIT','CANCELLED') else 2
    except Exception as exc:
        result = dict(loop.state) if loop else dict(id=args.output.name, mode='live', engine='movej', completed_cycles=0, config=vars(c))
        result.update(state='FAILED', reason=str(exc), finished_unix=time.time())
        if str(exc) == 'task_time_limit':
            result['state'] = 'TIMEOUT'
        if runner:
            stopped = runner.stop()
            result['stop'] = stopped
            if (str(exc).startswith('paused_') or str(exc) == 'pause_hold_not_confirmed'
                    or not (stopped.get('no_ownership_acquired') or stopped.get('stationary_confirmed'))):
                result['state'] = 'STOP_UNCONFIRMED'
        if supervision.diagnostics:
            result['connection_diagnostics'] = supervision.diagnostics
            save_json(args.output/'connection_diagnostics.json', supervision.diagnostics)
        save_json(args.output/'status.json', result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        return 2
    finally:
        watchdog_stop.set()
        if watcher:
            watcher.join(timeout=2)
        if io:
            for key, value in io.operation_counts.items():
                operation_totals[key] = operation_totals.get(key, 0)+value
            save_json(args.output/'operations.json', operation_totals)
            io.close()
        if policy:
            policy.close()
        lock.close();replay_lock.close()


if __name__ == '__main__':
    raise SystemExit(main())
