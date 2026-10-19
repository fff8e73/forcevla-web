import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from robot_control.planner_loop import LoopRunner, PlannerLoop, adapt, prepare_waypoints, profile
from test_first10_replay import Clock, IO, Solver


class Geometry(Solver):
    def poses(self, q):
        p=np.zeros((2,6));p[:,0]=[q[0],q[7]];return p


class Plant(IO):
    def __init__(self, clock):
        super().__init__(clock)
        self.g.update(target=[0.,0.],actual=[0.,0.],calibration=[[0.,1.6],[0.,1.6]],rates={'fine':.03})
        self.session='fake-owned-session'
        self.after_move=lambda:None
        self.allowed_targets=None
        self.grips=[]
    def read(self):
        f=super().read();f['tcp']=Geometry().poses(self.q).tolist();return f
    def set_grippers(self, values):
        self.grips.append(np.array(values))
        self.g.update(target=list(values),actual=list(values),applied_target=list(values))
    def move(self, q):
        super().move(q);self.after_move()


class Source:
    def __init__(self, io):self.io=io;self.reads=0;self.repeat=False;self.barriers=[]
    def read_after(self, barrier, previous_id):
        self.reads+=1;self.barriers.append(barrier)
        q=self.io.q.copy();state=np.zeros(26,np.float32);state[:6],state[13:19]=Geometry().poses(q)
        return SimpleNamespace(id='same' if self.repeat else 'observation-'+str(self.reads),
                               joints=q, state=state, images={},metadata={}, observed_at=time.monotonic(),
                               source_times={'image':time.time(),'joints':time.time()})


class Policy:
    def __init__(self):self.calls=[];self.bad=False;self.delay=0
    def predict(self, raw, request_id):
        self.calls.append((request_id,raw['state'].copy()))
        time.sleep(self.delay)
        actions=np.tile(np.r_[raw['state'][:7],raw['state'][13:20]],(1,50,1))
        actions[0,:,0]+=np.arange(1,51)*.0001
        actions[0,:,7]+=np.arange(1,51)*.0001
        actions[0,:,6]=-.001
        actions[0,:,13]=.005
        if self.bad:actions[0,0,0]=np.nan
        return dict(actions=actions)


def planner(actions, observation, output):
    q=np.tile(observation.joints,(len(actions),1))
    q[:,0]=actions[:,0];q[:,7]=actions[:,7]
    return q


class LoopTests(unittest.TestCase):
    def make(self, root, *, cycles=2, steps=20):
        clock=Clock();io=Plant(clock);cancel=threading.Event()
        runner=LoopRunner(io,Geometry(),None,None,np.zeros(14),clock=clock,sleep=clock.sleep,cancel=cancel.is_set)
        source=Source(io);policy=Policy()
        c=profile(dict(max_cycles=cycles,execution_horizon=steps,observation_timeout_s=.08))
        loop=PlannerLoop(source=source,policy=policy,planner=planner,runner=runner,config=c,output=root,
                         prompt='test',cancel=cancel.is_set)
        return loop,io,runner,source,policy,cancel
    def test_two_cycles_new_observations_and_exactly_twenty(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,source,policy,cancel=self.make(root)
            result=loop.run()
            self.assertEqual(result['state'],'COMPLETED_LIMIT',result)
            self.assertEqual(result['completed_cycles'],2)
            self.assertEqual(len(io.moves),42)
            self.assertEqual(len(policy.calls),2)
            self.assertGreater(policy.calls[1][1][0],policy.calls[0][1][0])
            self.assertEqual([h['executed_steps'] for h in result['history']],[20,20])
            self.assertTrue(result['stop']['confirmed'])
            self.assertEqual(io.selected,0)
            self.assertFalse(io.g['reserved'])
            self.assertAlmostEqual(io.grips[-1][1],.005)
            for row in result['history']:
                self.assertGreater(min(row['source_times'].values()),row['hold_before']['unix'])
    def test_repeated_observation_cannot_start_second_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,source,policy,cancel=self.make(root);source.repeat=True
            result=loop.run()
            self.assertEqual(result['completed_cycles'],1)
            self.assertEqual(len(policy.calls),1)
            self.assertEqual(len(io.moves),22)
            self.assertIn('new_observation_timeout',result['reason'])
            self.assertTrue(result['stop']['confirmed'])
    def test_bad_prediction_stops_without_motion_targets(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,s,policy,cancel=self.make(root);policy.bad=True
            result=loop.run()
            self.assertEqual(result['state'],'FAILED')
            self.assertEqual(len(io.moves),2)
            self.assertTrue(result['stop']['confirmed'])
    def test_cancel_during_motion_no_remaining_queue_consumption(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,s,policy,cancel=self.make(root)
            io.after_move=lambda:cancel.set() if len(io.moves)==8 else None
            result=loop.run()
            self.assertEqual(result['state'],'CANCELLED')
            self.assertEqual(len(io.moves),8)
            self.assertEqual(len(policy.calls),1)
            self.assertTrue(result['stop']['confirmed'])
    def test_cancel_during_rpc_discards_late_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,s,policy,cancel=self.make(root);policy.delay=.1
            timer=threading.Timer(.04,cancel.set);timer.start()
            result=loop.run();timer.join();time.sleep(.12)
            self.assertEqual(result['state'],'CANCELLED')
            self.assertEqual(len(io.moves),2)
            self.assertTrue(result['stop']['confirmed'])
    def test_preparation_failure_cannot_move(self):
        with tempfile.TemporaryDirectory() as root:
            loop,io,r,s,p,c=self.make(root)
            loop.planner=lambda *a:(_ for _ in ()).throw(RuntimeError('unreachable'))
            result=loop.run()
            self.assertEqual(result['state'],'FAILED')
            self.assertEqual(len(io.moves),2)
    def test_exact_twenty_and_tail_not_consumed(self):
        a=np.zeros((1,50,14));a[0,20:,:]=999
        bounded,_=adapt(a)
        self.assertEqual(bounded.shape,(20,14))
        self.assertFalse(bounded.any())
        for n in (0,51,-1,2.5,True):
            with self.assertRaises(ValueError):profile(dict(execution_horizon=n))
    def test_gripper_feedback_error_does_not_block_twenty_step_cycles(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, runner, source, policy, cancel = self.make(root)
            original_set = io.set_grippers

            def target_accepted_with_position_error(values):
                original_set(values)
                # Reproduce a motor error greater than the former 0.05 rad gate.
                io.g['actual'] = (np.asarray(values) + [-.0211003656, .0330214761]).tolist()

            io.set_grippers = target_accepted_with_position_error
            result = loop.run()
            self.assertEqual(result['state'], 'COMPLETED_LIMIT', result)
            self.assertEqual(result['completed_cycles'], 2)
            self.assertEqual(len(io.moves), 42)
            self.assertTrue(result['stop']['confirmed'])
            saved = json.loads((Path(root)/'cycles/0002/reached.json').read_text())
            self.assertEqual(len(saved), 20)
            last = saved[-1]
            error = abs(last['measured']['gripper']['actual'][1] - last['gripper_target'][1]) * 1.6
            self.assertGreater(error, .05)

    def test_gripper_slew_does_not_add_per_waypoint_wait(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, runner, source, policy, cancel = self.make(root)
            original_set = io.set_grippers

            def target_accepted_before_slew_finishes(values):
                original_set(values)
                io.g['applied_target'] = [0., 0.]
                io.g['actual'] = [0., 0.]

            io.set_grippers = target_accepted_before_slew_finishes
            result = loop.run()
            self.assertEqual(result['state'], 'COMPLETED_LIMIT', result)
            self.assertEqual(result['completed_cycles'], 2)
            self.assertEqual(len(io.moves), 42)
            self.assertTrue(result['stop']['confirmed'])

    def test_configured_prefix_is_consumed_in_each_new_cycle(self):
        for steps in (1, 7, 20, 50):
            with self.subTest(steps=steps), tempfile.TemporaryDirectory() as root:
                loop, io, runner, source, policy, cancel = self.make(root, steps=steps)
                result = loop.run()
                self.assertEqual(result['state'], 'COMPLETED_LIMIT', result)
                self.assertEqual(len(io.moves), 2 + 2*steps)
                self.assertEqual([h['executed_steps'] for h in result['history']], [steps, steps])
                self.assertEqual([h['execution_horizon'] for h in result['history']], [steps, steps])
                self.assertEqual(result['config']['execution_horizon'], steps)
                self.assertTrue(result['stop']['confirmed'])

    def test_defaults_and_selected_gripper_correction(self):
        self.assertEqual(profile({}).max_cycles, 50)
        self.assertEqual(profile({}).execution_horizon, 20)
        actions = np.zeros((1, 50, 14))
        actions[0, 7:, [6, 13]] = 2
        selected, _ = adapt(actions, execution_horizon=7)
        self.assertEqual(selected.shape, (7, 14))
        with self.assertRaisesRegex(Exception, 'gripper_correction_excessive'):
            adapt(actions, execution_horizon=8)

    def test_wrong_prepared_count_is_rejected_before_motion(self):
        with tempfile.TemporaryDirectory() as root:
            loop, io, *_ = self.make(root, steps=7)
            loop.planner = lambda *args: np.zeros((8, 14))
            result = loop.run()
            self.assertEqual(result['state'], 'FAILED')
            self.assertEqual(result['reason'], 'prepared_waypoint_count_or_values_invalid')
            self.assertEqual(len(io.moves), 2)

    def test_partial_qp_batches_do_not_add_padding_waypoints(self):
        class FakeQP(Geometry):
            parameters = {'collision_minimum_distance': .005}
            def __init__(self): self.lengths = []
            def rollout(self, actions, seed, state, output, *, prefix_steps=50):
                self.assert_input(actions, prefix_steps)
                self.lengths.append(prefix_steps)
                q = np.tile(seed, (prefix_steps, 1))
                q[:, 0] = actions[:prefix_steps, 0]
                q[:, 7] = actions[:prefix_steps, 7]
                np.savez_compressed(Path(output)/'qp_trajectory.npz', joint_positions=q,
                                    target_indices=np.arange(prefix_steps))
                return {'status': 'passed_scoped_qp', 'reasons': []}
            def assert_input(self, actions, prefix_steps):
                assert actions.shape == (50, 14)
                assert 1 <= prefix_steps <= 50
        for count in (1, 4, 7, 20, 49, 50):
            with self.subTest(steps=count), tempfile.TemporaryDirectory() as root:
                solver = FakeQP()
                actions = np.zeros((count, 14))
                actions[:, 0] = np.arange(count)*.0001
                actions[:, 7] = np.arange(count)*.0001
                observation = SimpleNamespace(joints=np.zeros(14), state=np.zeros(26))
                result = prepare_waypoints(solver, actions, observation, Path(root)/'plan')
                self.assertEqual(result.shape, (count, 14))
                np.testing.assert_array_equal(result[:, 0], actions[:, 0])
                self.assertEqual(sum(solver.lengths), 10*count)
                self.assertEqual(solver.lengths[-1], (count%5 or 5)*10)

    def test_force_is_diagnostic_only(self):
        c=Clock();io=Plant(c);r=LoopRunner(io,Geometry(),None,None,np.zeros(14),clock=c,sleep=c.sleep)
        r.baseline_wrench=np.full((2,6),100.)
        self.assertEqual(r.sample(0)['raw_wrench_delta'],[[-100.]*6]*2)

if __name__=='__main__':unittest.main()
