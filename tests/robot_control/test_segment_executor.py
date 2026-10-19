import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.prepared_joint_plan import PreparedJointPlan
from robot_control.segment_executor import BufferedSandboxBackend, SegmentExecutor, SegmentLimits
from test_joint_execution import FakeIO


def synthetic_plan(seed, observed, amplitude=.001, n=100, dt=.002):
    u=np.arange(1,n+1)/n
    curve=10*u**3-15*u**4+6*u**5
    q=np.tile(seed,(n,1))
    q[:,0]+=amplitude*curve
    q[:,7]-=amplitude*curve
    return PreparedJointPlan('synthetic-small-motion', observed, seed.copy(),q,
                             np.full((n,2),.5),dt,{'scope':'synthetic_not_model'})


class SegmentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now=10.
        self.io=FakeIO(self.tmp.name,lambda:self.now)
        self.b=BufferedSandboxBackend(self.io,np.tile([-2.,2.],(14,1)),np.ones(14),
                                      Path(self.tmp.name)/'execution.json',clock=lambda:self.now)
        self.token=self.b.acquire('test')
        self.e=SegmentExecutor(self.b,clock=lambda:self.now)
        self.plan=synthetic_plan(np.zeros(14),self.now)

    def step(self):
        self.now=self.e.next_due
        self.b.heartbeat(self.token)
        self.e.tick()

    def settle(self):
        for _ in range(20):
            self.now+=.02
            self.e.poll_hold()

    def test_normal_prefix_and_terminal_hold(self):
        self.e.start(self.plan,self.token)
        for _ in range(101):self.step()
        self.assertEqual(self.e.reason,'segment_complete')
        self.assertEqual(self.e.index,100)
        self.assertIsNone(self.e.plan)
        self.settle()
        self.assertEqual(self.e.state,'HOLD_CONFIRMED')
        np.testing.assert_array_equal(self.io.q,self.plan.joints[-1])
        self.assertEqual(self.b.grip_target.tolist(),[.5,.5])
        self.assertFalse(self.e.report()['hardware_execution_verified'])

    def test_deadline_miss_never_catches_up(self):
        self.e.start(self.plan,self.token)
        self.step()
        self.now=self.e.next_due+.01
        self.e.tick()
        self.assertEqual(self.e.reason,'segment_deadline_missed')
        count=len(self.io.events)
        self.now+=.1
        self.e.tick()
        self.assertEqual(len(self.io.events),count)
        self.assertEqual(self.e.index,1)

    def test_manual_abort_retains_last_goal(self):
        self.e.start(self.plan,self.token)
        for _ in range(30):self.step()
        held=self.io.q.copy()
        self.e.request_abort()
        self.e.tick()
        self.settle()
        np.testing.assert_array_equal(self.io.q,held)
        self.assertEqual(self.e.reason,'manual_abort')
        self.assertEqual(self.e.state,'HOLD_CONFIRMED')

    def test_upstream_lease_expires_despite_motion_ticks(self):
        self.b.guard.lease_s=.04
        self.b.heartbeat(self.token)
        self.e.start(self.plan,self.token)
        for _ in range(25):
            if self.e.state!='RUNNING':break
            self.now=self.e.next_due
            self.e.tick()
        self.assertNotEqual(self.e.state,'RUNNING')
        self.assertLess(self.e.index,25)
        self.assertEqual(self.b.stop_report['reason'],'upstream_lease_expired')

    def test_partial_failure_no_remaining_motion(self):
        self.e.start(self.plan,self.token)
        for _ in range(20):self.step()
        self.io.fail_side='right'
        self.step()
        self.assertEqual(self.e.state,'FAULT_UNCONFIRMED')
        self.assertEqual(self.io.gate.state,'held')
        self.assertTrue(self.io.gate.reserved)
        self.assertIsNone(self.e.plan)

    def test_feedback_failure_discards_queue(self):
        self.e.start(self.plan,self.token)
        self.io.stale=True
        self.now=self.e.next_due
        self.e.tick()
        self.assertEqual(self.e.state,'FAULT_UNCONFIRMED')
        self.assertEqual(self.e.index,0)

    def test_invalid_paths_rejected_before_any_motion(self):
        cases=[('segment_displacement',lambda p:p.joints.__setitem__((-1,0),.1)),
               ('segment_acceleration',lambda p:p.joints.__setitem__((50,0),p.joints[50,0]+.00002)),
               ('fixed_grippers',lambda p:p.grippers.__setitem__((50,0),.6)),
               ('segment_seed',lambda p:p.seed.__setitem__(0,.1))]
        for reason,change in cases:
            with self.subTest(reason=reason):
                p=synthetic_plan(np.zeros(14),self.now)
                change(p)
                with self.assertRaisesRegex(ControlFault,reason):self.e.start(p,self.token)
        self.assertFalse(any(e[0]=='arm' for e in self.io.events))

    def test_stale_observation_and_unfinishable_age_rejected(self):
        for observed in [5.,7.6]:
            p=synthetic_plan(np.zeros(14),observed)
            with self.assertRaisesRegex(ControlFault,'observation_expired'):
                self.e.start(p,self.token)

    def test_terminal_hold_acceleration_checked(self):
        p=synthetic_plan(np.zeros(14),self.now)
        p.joints[:,0]=np.arange(1,101)*.00001
        with self.assertRaisesRegex(ControlFault,'acceleration_or_terminal_hold'):
            self.e.start(p,self.token)

    def test_caller_mutation_cannot_change_accepted_path(self):
        self.e.start(self.plan,self.token)
        expected=self.plan.joints[0].copy()
        self.plan.joints[:]=1.
        self.step()
        np.testing.assert_array_equal(self.io.q,expected)

    def test_production_cannot_enable_moving_executor(self):
        self.io.production=True
        with self.assertRaisesRegex(ControlFault,'sandbox_only'):
            SegmentExecutor(self.b)

    def test_late_submission_trips_before_next_tick(self):
        original=self.io.publish_arm
        def delayed(side,q):
            original(side,q)
            self.now+=.001
        self.io.publish_arm=delayed
        self.e.start(self.plan,self.token)
        self.step()
        self.assertEqual(self.e.reason,'segment_submission_overrun')
        self.assertEqual(self.e.index,1)

    def test_zero_time_repeated_tick_does_not_duplicate(self):
        self.e.start(self.plan,self.token)
        self.step()
        self.e.tick()
        self.assertEqual(self.e.index,1)

    def test_no_restart_after_abort(self):
        self.e.start(self.plan,self.token)
        self.e.request_abort()
        self.e.tick()
        with self.assertRaisesRegex(ControlFault,'single_use'):
            self.e.start(self.plan,self.token)

    def test_log_capacity_fault_does_not_send_next_target(self):
        self.e.start(self.plan,self.token)
        self.b.events.extend([{}]*20000)
        self.step()
        self.assertEqual(self.e.reason,'tick_log_capacity')
        self.assertEqual(self.e.index,0)

    def test_abnormal_hold_budget_is_checked_at_every_tick(self):
        e=SegmentExecutor(self.b,SegmentLimits(max_abnormal_hold_acceleration_rad_s2=.1),clock=lambda:self.now)
        with self.assertRaisesRegex(ControlFault,'abnormal_hold_budget'):
            e.start(self.plan,self.token)


class NativeSegmentTests(unittest.TestCase):
    def test_native_qp_prefix_runs_without_retiming(self):
        import json
        from robot_control.common import ROOT
        from robot_control.native_qp import NativeQP
        from robot_control.prepared_joint_plan import prepare_native_plan
        limits=json.loads((ROOT/'config/planning.json').read_text())
        solver=NativeQP(limits['expected_qp_parameters'],limits)
        seed=np.array([.69,-1.36,-.75,-.90,-.23,.24,-1.05,
                       -1.29,-1.09,1.14,-1.26,.58,-.19,-.035])
        poses=solver.poses(seed)
        state=np.r_[poses[0],.5,np.zeros(6),poses[1],.5,np.zeros(6)]
        actions=np.tile(np.r_[poses[0],.5,poses[1],.5],(1,50,1))
        actions[0,:,0]+=np.linspace(0,.00001,50)
        now=[10.]
        with tempfile.TemporaryDirectory() as directory:
            p=prepare_native_plan(solver,actions=actions,seed=seed,state=state,
                request_id='native-small',observed_at=10.,output=Path(directory)/'plan',
                prefix_steps=3,max_correction=.05,clock=lambda:now[0])
            io=FakeIO(directory,lambda:now[0])
            io.q[:]=seed
            io.target[:]=seed
            b=BufferedSandboxBackend(io,solver.bounds,solver.velocity,
                Path(directory)/'execution.json',clock=lambda:now[0])
            token=b.acquire('native-test')
            e=SegmentExecutor(b,clock=lambda:now[0])
            e.start(p,token)
            for _ in range(len(p.joints)+1):
                now[0]=e.next_due
                b.heartbeat(token)
                e.tick()
            self.assertEqual(e.reason,'segment_complete')
            self.assertEqual(e.index,50)
            np.testing.assert_allclose(io.q,p.joints[-1],rtol=0,atol=1e-12)
            np.testing.assert_allclose(np.diff([t['scheduled'] for t in e.trace]),.002,rtol=0,atol=1e-12)
