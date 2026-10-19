import copy
import tempfile
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.planner_loop import LoopRunner, PlannerLoop
from robot_control.planner_recovery import require_unchanged_hold, run_recovering, wait_for_recovery
from robot_control.planner_supervision import PlannerSupervision
import test_planner_loop as fixtures
from test_planner_loop import Geometry


class RecoveryTests(unittest.TestCase):
    def test_orchestrator_auto_resumes_same_task_after_confirmed_recovery(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, runner, source, policy, cancelled = fixtures.LoopTests().make(root, cycles=2, steps=4)
            clock = SimpleNamespace(now=100.)
            sup = PlannerSupervision(loop.c, clock=lambda: clock.now)
            loop.reason = sup.reason
            loop.pause = lambda: bool(sup.pause_reason)
            def dropout():
                if len(io.moves) == 8:
                    clock.now = 106.; sup.heartbeat(100., {'active_connections': 1})
                    cancelled.set()
            io.after_move = dropout
            replacements = []
            def make_loop(prior, held, attempt):
                require_unchanged_hold(io.read(), held)
                cancelled.clear(); io.after_move = lambda: None
                replacements.append(attempt)
                replacement = LoopRunner(io, Geometry(), None, None, io.q.copy(),
                                         clock=runner.clock, sleep=runner.sleep, cancel=cancelled.is_set)
                return PlannerLoop(source=source, policy=policy, planner=loop.planner,
                                   runner=replacement, config=loop.c, output=root, prompt='test',
                                   cancel=cancelled.is_set, resume_state=prior,
                                   started=loop.started, attempt=attempt)
            actual_wait = wait_for_recovery
            def heartbeat_tick(_):
                clock.now += 1.05
                sup.heartbeat(clock.now, {'active_connections': 1})
            def wait(current, supervisor, **kw):
                return actual_wait(current, supervisor, clock=lambda: clock.now, sleep=heartbeat_tick)
            with patch('robot_control.planner_recovery.wait_for_recovery', side_effect=wait):
                result = run_recovering(loop, sup, make_loop)
            self.assertEqual(result['state'], 'COMPLETED_LIMIT', result)
            self.assertEqual(result['completed_cycles'], 2)
            self.assertEqual(replacements, [2])
            self.assertEqual(len(result['pauses']), 1)
            self.assertEqual(len(policy.calls), 3)

    def paused_loop(self, root, *, rpc=False):
        loop, io, runner, source, policy, cancelled = fixtures.LoopTests().make(root, cycles=2, steps=4)
        loop.pause = lambda: True
        loop.reason = lambda: 'browser_heartbeat_timeout'
        if rpc:
            def late_prediction(*args):
                cancelled.set()
                time.sleep(.1)
                return dict(actions=np.zeros((1, 50, 14)))
            policy.predict = late_prediction
        else:
            io.after_move = lambda: cancelled.set() if len(io.moves) == 8 else None
        result = loop.run()
        self.assertEqual(result['state'], 'PAUSED', result)
        self.assertTrue(result['stop']['confirmed'])
        return loop, io, runner, source, policy, cancelled

    def test_pause_discards_motion_tail_and_new_attempt_observes_current_state(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, runner, source, policy, cancel = self.paused_loop(root)
            self.assertEqual(loop.state['completed_cycles'], 1)
            stopped_count = len(io.moves)
            self.assertEqual(stopped_count, 8)  # 2 acquisition + 4 + 2 interrupted.
            held = copy.deepcopy(loop.state['stop']['final_feedback'])
            cancel.clear(); io.after_move = lambda: None
            require_unchanged_hold(io.read(), held)
            replacement = LoopRunner(io, Geometry(), None, None, io.q.copy(),
                                     clock=runner.clock, sleep=runner.sleep, cancel=cancel.is_set)
            resumed = PlannerLoop(source=source, policy=policy, planner=loop.planner,
                                  runner=replacement, config=loop.c, output=root, prompt='test',
                                  cancel=cancel.is_set, resume_state=loop.state,
                                  started=loop.started, attempt=2)
            result = resumed.run()
            self.assertEqual(result['state'], 'COMPLETED_LIMIT', result)
            self.assertEqual(result['completed_cycles'], 2)
            self.assertEqual(len(policy.calls), 3)
            self.assertGreater(policy.calls[2][1][0], policy.calls[1][1][0])
            self.assertEqual(len(io.moves)-stopped_count, 6)  # reacquire + new full prefix
            self.assertTrue((Path(root)/'cycles/0002-attempt-2/observation.npz').is_file())
            self.assertEqual(len(result['history']), 2)

    def test_ordinary_cancel_is_terminal_and_keeps_true_reason(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, r, s, p, cancel = fixtures.LoopTests().make(root)
            loop.reason = lambda: 'user_cancelled'
            io.after_move = lambda: cancel.set() if len(io.moves) == 4 else None
            result = loop.run()
            self.assertEqual(result['state'], 'CANCELLED')
            self.assertEqual(result['reason'], 'user_cancelled')

    def test_pause_during_rpc_discards_late_result(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, *_ = self.paused_loop(root, rpc=True)
            self.assertEqual(len(io.moves), 2)
            loop.active_future.result(timeout=1)
            self.assertEqual(len(io.moves), 2)

    def test_hold_must_remain_unchanged(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, *_ = self.paused_loop(root)
            held = loop.state['stop']['final_feedback']
            for change in (
                lambda f: f.update(input=1),
                lambda f: f.update(mode=0),
                lambda f: f['joints'].__setitem__(0, .5),
                lambda f: f['velocities'].__setitem__(0, .1),
                lambda f: f['joints'].__setitem__(0, float('nan')),
                lambda f: f['gripper'].update(reserved=True),
                lambda f: f['gripper'].update(target=[1., 1.]),
                lambda f: f['gripper'].update(boot_id='another-boot'),
                lambda f: f['gripper'].update(healthy=False),
            ):
                f = io.read(); change(f)
                with self.assertRaisesRegex(ControlFault, 'paused_robot_state_changed'):
                    require_unchanged_hold(f, held)

    def test_failed_hold_can_never_wait_for_resume(self):
        with tempfile.TemporaryDirectory() as root:
            loop, *_ = self.paused_loop(root)
            loop.state['stop']['confirmed'] = False
            with self.assertRaisesRegex(ControlFault, 'pause_hold_not_confirmed'):
                wait_for_recovery(loop, None)

    def test_manual_stop_while_waiting_never_resumes(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, *_ = self.paused_loop(root)
            sup = PlannerSupervision(loop.c)
            sup.cancel('user_cancelled')
            count = len(io.moves)
            self.assertFalse(wait_for_recovery(loop, sup))
            self.assertEqual(loop.state['reason'], 'user_cancelled')
            self.assertEqual(len(io.moves), count)

    def test_ready_connection_waits_for_late_work_without_device_commands(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, *_ = self.paused_loop(root)
            future = loop.active_future = Future()
            clock = SimpleNamespace(now=100.)
            sup = PlannerSupervision(loop.c, clock=lambda: clock.now)
            clock.now = 106.; sup.heartbeat(100., {'active_connections': 1}); sup.interrupted()
            count = len(io.moves)
            ticks = []
            def sleep(_):
                clock.now += 1.05
                sup.heartbeat(clock.now, {'active_connections': 1})
                ticks.append(clock.now)
                if len(ticks) >= 6:
                    future.set_result('discarded')
            self.assertTrue(wait_for_recovery(loop, sup, clock=lambda: clock.now, sleep=sleep))
            self.assertGreaterEqual(len(ticks), 6)
            self.assertEqual(len(io.moves), count)


if __name__ == '__main__':
    unittest.main()
