"""Wait for recovery only after confirmed hold; discard interrupted work."""
import time

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import save_json


def require_unchanged_hold(feedback, held):
    """Never retake control after movement, source changes or another claim."""
    g, previous = feedback['gripper'], held['gripper']
    q, v = np.asarray(feedback['joints']), np.asarray(feedback['velocities'])
    target = np.asarray(g['target'])
    if (feedback['input'] != 0 or feedback['mode'] != 1
            or q.shape != (14,) or v.shape != (14,)
            or not np.isfinite(q).all() or not np.isfinite(v).all()
            or np.max(np.abs(v)) > .02
            or np.max(np.abs(q-held['joints'])) > .002
            or g.get('reserved') or g.get('boot_id') != previous.get('boot_id')
            or g.get('healthy') is False
            or target.shape != (2,) or not np.isfinite(target).all()
            or not np.allclose(g['target'], previous['target'], atol=1e-6, rtol=0)):
        raise ControlFault('paused_robot_state_changed')


def wait_for_recovery(loop, supervisor, *, clock=time.monotonic, sleep=time.sleep, log=lambda *a, **k: None):
    report = loop.state.get('stop', {})
    if report.get('no_ownership_acquired'):
        feedback = loop.runner.io.read()
        require_unchanged_hold(feedback, feedback)
        report['final_feedback'] = feedback
    if not report.get('confirmed') or not report.get('final_feedback'):
        raise ControlFault('pause_hold_not_confirmed')
    held = report['final_feedback']
    paused_at = clock()
    last_write = float('-inf')
    while True:
        supervisor.interrupted()
        if supervisor.terminal_reason:
            loop.update('TIMEOUT' if supervisor.reason() == 'task_time_limit' else 'CANCELLED',
                        reason=supervisor.reason(), finished_unix=time.time())
            return False
        try:
            feedback = loop.runner.io.read()
        except Exception as exc:
            raise ControlFault('paused_feedback_unavailable:' + str(exc)) from exc
        require_unchanged_hold(feedback, held)
        # A cancelled RPC or solver may finish late. Drain it before new work;
        # its output is never consumed or submitted to the robot.
        idle = loop.active_future is None or loop.active_future.done()
        recovery = supervisor.recovery()
        if clock()-last_write >= .2:
            loop.update('PAUSED', recovery=dict(recovery, stale_work_finished=idle),
                        pause_elapsed_s=clock()-paused_at)
            last_write = clock()
        if idle and recovery['ready'] and supervisor.resume():
            log('browser_recovered', paused_s=clock()-paused_at)
            loop.update('RESUMING', reason=None, recovery=recovery)
            return True
        sleep(.05)


def run_recovering(loop, supervisor, make_loop, *, log=lambda *a, **k: None):
    pauses = []
    attempt = 1
    while True:
        result = loop.run()
        result['pauses'] = pauses
        result['connection_diagnostics'] = supervisor.diagnostics or dict(
            captured_unix=time.time(), connection_health=supervisor.health)
        save_json(loop.output/'status.json', result)
        if result['state'] != 'PAUSED':
            return result
        pauses.append(dict(attempt=attempt, cycle=result['cycle'],
                           executed_steps=result.get('execution', {}).get('executed_model_steps', 0),
                           diagnostics=supervisor.diagnostics))
        log('paused_after_confirmed_hold', attempt=attempt, reason=result['reason'])
        if not wait_for_recovery(loop, supervisor, log=log):
            return loop.state
        held = result['stop']['final_feedback']
        attempt += 1
        loop = make_loop(dict(loop.state), held, attempt)
