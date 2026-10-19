import copy
import unittest
import numpy as np
from robot_control.command_bridge import ControlFault
from robot_control.first10_replay import ReplayRunner, check_start, load_targets


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, t): self.now += t


class Solver:
    bounds = np.tile([-3., 3.], (14, 1))
    def collisions(self, q): return [dict(distance_m=.1)]


class IO:
    velocity = np.full(14, .2)
    def __init__(self, clock):
        self.clock = clock
        self.q = np.zeros(14)
        self.selected = 0
        self.g = dict(reserved=False, state='idle', applied_target=[0., 0.])
        self.moves = []
        self.fail_at = None
        self.read_fault = None
        self.selections = []
    def read(self):
        if self.read_fault: raise ControlFault(self.read_fault)
        return dict(input=self.selected, mode=1, joints=self.q.tolist(), velocities=[0.]*14,
                    joint_target=self.q.tolist(), planner_target=self.q.tolist(),
                    tcp=[[0.]*6]*2, wrench=[[0.]*6]*2, sample_time=self.clock(),
                    gripper=copy.deepcopy(self.g))
    def input(self): return self.selected
    def gripper(self): return copy.deepcopy(self.g)
    def owns(self, g): return g['reserved']
    def claim(self): self.g.update(reserved=True, state='active')
    def heartbeat(self): pass
    def hold(self): self.g['state'] = 'held'
    def release(self): self.g.update(reserved=False, state='idle')
    def select(self, index): self.selections.append(index); self.selected=index
    def move(self, q):
        self.moves.append(np.array(q))
        if len(self.moves) == self.fail_at: raise ControlFault('movej_timeout')
        self.q = np.array(q)


class ReplayTests(unittest.TestCase):
    def setup_runner(self):
        clock = Clock()
        io = IO(clock)
        runner = ReplayRunner(io, Solver(), np.zeros((10,14)),
                              np.array([np.full(14, .005*(i+1)) for i in range(10)]),
                              np.zeros(14), clock=clock, sleep=clock.sleep)
        return runner, io
    def test_full_ten_and_stop(self):
        r, io = self.setup_runner()
        r.run()
        stop=r.stop()
        self.assertEqual(len(r.completed),10)
        self.assertEqual(len(io.moves),12)  # seed before/after mux selection + ten goals
        self.assertTrue(stop['stationary_confirmed'])
        self.assertTrue(stop['gripper_released'])
        self.assertEqual(io.selections,[2,0])
    def test_rpc_failure_never_issues_next_goal(self):
        r, io = self.setup_runner(); io.fail_at=5
        with self.assertRaisesRegex(ControlFault,'movej_timeout'): r.run()
        r.stop()
        self.assertEqual(len(io.moves),5)
        self.assertEqual(len(r.completed),2)
        self.assertEqual(io.selected,0)
    def test_stale_feedback_still_disconnects(self):
        r, io = self.setup_runner()
        r.claimed=r.selected=True; io.claim(); io.selected=2; io.read_fault='stale'
        stop=r.stop()
        self.assertEqual(io.selected,0)
        self.assertFalse(stop['stationary_confirmed'])
        self.assertTrue(io.g['reserved'])
    def test_other_source_not_overwritten(self):
        r, io = self.setup_runner()
        r.claimed=r.selected=True; io.claim(); io.selected=1
        stop=r.stop()
        self.assertFalse(stop['input_none'])
        self.assertEqual(io.selections,[])
    def test_changed_seed_rejected_before_any_command(self):
        r, io = self.setup_runner(); io.q[0]=.02
        with self.assertRaisesRegex(ControlFault,'seed_changed'): r.run()
        r.stop()
        self.assertEqual(io.moves,[])
        self.assertFalse(io.g['reserved'])
    def test_cancel_before_start(self):
        r, io = self.setup_runner(); r.cancel=lambda:True
        with self.assertRaisesRegex(ControlFault,'operator_cancelled'): r.run()
        self.assertEqual(io.moves,[])
    def test_gripper_change_rejected(self):
        r, io = self.setup_runner(); io.g['applied_target']=[.1,0]
        with self.assertRaisesRegex(ControlFault,'fixed_gripper'): r.run()
        self.assertEqual(io.moves,[])
    def test_source_identity_and_exact_ten(self):
        raw, actions, state, q, meta=load_targets()
        self.assertEqual(raw.shape,(10,14))
        self.assertTrue(np.array_equal(actions[:,[6,13]],np.zeros((10,2))))
        self.assertEqual(meta['request_id'],'97c9b0ed7a034509a87c75829b88806e')
        check_start(q,np.stack([state[:6],state[13:19]]),q,state)
        with self.assertRaisesRegex(ControlFault,'recorded_start'):
            check_start(q+.01,np.stack([state[:6],state[13:19]]),q,state)
    def test_raw_wrench_change_is_logged_without_invented_stop(self):
        r, io = self.setup_runner()
        r.baseline_wrench = np.full((2, 6), 100.)
        f = r.sample(0)
        self.assertEqual(f['raw_wrench_delta'], [[-100.] * 6] * 2)

    def test_stop_is_idempotent(self):
        r, io = self.setup_runner(); r.run(); r.stop(); r.stop()
        self.assertEqual(io.selections,[2,0])

if __name__=='__main__': unittest.main()
