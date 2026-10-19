"""Production supervision logic, without ROS or actuators."""
import unittest
from types import SimpleNamespace

from robot_control.planner_supervision import PlannerSupervision


class SupervisionTests(unittest.TestCase):
    def make(self):
        clock = SimpleNamespace(now=100.)
        s = PlannerSupervision(SimpleNamespace(max_task_duration_s=600., browser_disconnect_timeout_s=5.),
                               clock=lambda: clock.now, wall=lambda: clock.now+1000)
        return s, clock

    def paused(self):
        s, c = self.make()
        c.now = 105.01
        s.heartbeat(100., {'state': 'open', 'active_connections': 1})
        self.assertTrue(s.interrupted())
        self.assertEqual(s.reason(), 'browser_heartbeat_timeout')
        self.assertIsNone(s.terminal_reason)
        self.assertGreater(s.diagnostics['browser_heartbeat_age_s'], 5)
        return s, c

    def recover(self, s, c):
        for value in (106., 107.05, 108.1, 109.15):
            c.now = value
            s.heartbeat(value, {'active_connections': 1})

    def test_timeout_is_recoverable_and_requires_distinct_stable_responses(self):
        s, c = self.paused()
        c.now = 106.; s.heartbeat(106., {'active_connections': 1})
        for value in (106.5, 107., 107.4):
            c.now = value; s.heartbeat(106., {'active_connections': 1})
        self.assertFalse(s.recovery()['ready'])
        self.recover(s, c)
        self.assertTrue(s.resume())
        self.assertFalse(s.interrupted())
        self.assertIsNone(s.pause_reason)

    def test_flapping_connection_resets_stability(self):
        s, c = self.paused()
        for value in (106., 107., 110.):
            c.now = value; s.heartbeat(value, {'active_connections': 1})
        self.assertFalse(s.resume())
        self.assertEqual(s.recovery()['stable_s'], 0)

    def test_manual_stop_cannot_auto_resume(self):
        s, c = self.paused(); s.cancel('user_cancelled'); self.recover(s, c)
        self.assertFalse(s.resume())
        self.assertEqual(s.reason(), 'user_cancelled')

    def test_parent_loss_is_terminal(self):
        s, c = self.make(); c.now = 102.1
        self.assertTrue(s.interrupted())
        self.assertEqual(s.reason(), 'parent_disconnected')

    def test_pause_does_not_reset_total_time_limit(self):
        s, c = self.paused(); c.now = 700.
        s.heartbeat(700., {'active_connections': 1}); s.interrupted()
        self.assertEqual(s.reason(), 'task_time_limit')
        self.assertFalse(s.resume())

    def test_invalid_heartbeat_is_terminal(self):
        for value in (float('nan'), float('inf'), 101., True, None):
            s, c = self.make(); s.heartbeat(value)
            self.assertEqual(s.reason(), 'invalid_browser_heartbeat')

    def test_no_active_owner_connection_cannot_resume(self):
        s, c = self.paused(); self.recover(s, c)
        s.heartbeat(c.now, {'active_connections': 0})
        self.assertFalse(s.resume())

    def test_recovery_wait_keeps_motion_interrupted_until_explicit_transition(self):
        s, c = self.paused(); self.recover(s, c)
        self.assertTrue(s.interrupted())
        self.assertTrue(s.recovery()['ready'])
        self.assertTrue(s.resume())


if __name__ == '__main__':
    unittest.main()
