import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.measured_hold import MeasuredHoldBackend
from test_joint_execution import FakeIO


class MeasuredTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now=10.
        self.io=FakeIO(self.tmp.name,lambda:self.now)
        self.io.input=2
        self.io.q[:]=.2
        self.io.target[:]=.3
        self.b=MeasuredHoldBackend(self.io,np.tile([-2.,2.],(14,1)),np.ones(14),
             Path(self.tmp.name)/'execution.json',clock=lambda:self.now)
        self.b.start_watchdog()
        self.addCleanup(self.b.close)

    def test_measured_seed_not_previous_target(self):
        self.b.acquire('test')
        np.testing.assert_array_equal(self.b.last_q,np.full(14,.2))
        self.assertEqual(self.io.input,3)
        self.io.q[:]=.201
        np.testing.assert_array_equal(self.b.last_q,np.full(14,.2))

    def test_reject_impedance_before_mutation(self):
        self.io.mode=3
        with self.assertRaises(ControlFault):self.b.acquire('test')
        self.assertEqual(self.io.events,[])

    def test_moving_rejected_before_mutation(self):
        self.io.v[:]=.1
        with self.assertRaises(ControlFault):self.b.acquire('test')
        self.assertEqual(self.io.events,[])

    def test_missing_target_not_fabricated(self):
        from dataclasses import replace
        original=self.io.feedback
        self.io.feedback=lambda:replace(original(),joint_target=np.full(14,np.nan),target_received_at=0.)
        self.b.acquire('test')
        f=self.b.checked_feedback()
        self.assertTrue(np.isnan(f.joint_target).all())
        self.b.trip('test')
        for _ in range(100):
            self.now+=.02
            self.b.poll()
        self.assertEqual(self.b.state,'FAULT_UNCONFIRMED')

    def test_displacement_trips(self):
        self.b.acquire('test')
        self.io.q[0]+=.006
        self.b.poll()
        self.assertEqual(self.b.state,'FAULT_UNCONFIRMED')
        self.assertEqual(self.io.gate.state,'held')


class MonitorOverrideTests(unittest.TestCase):
    def test_override_does_not_change_joint_limit_or_claim_request(self):
        with tempfile.TemporaryDirectory() as directory:
            now=[10.]
            io=FakeIO(directory,lambda:now[0])
            b=MeasuredHoldBackend(io,np.tile([-2.,2.],(14,1)),np.ones(14),
                Path(directory)/'execution.json',clock=lambda:now[0],
                gripper_monitor_limit_rad_s=.5)
            b.start_watchdog()
            try:
                token=b.acquire('test')
                self.assertEqual(b.stop_speed,.02)
                self.assertEqual(b.gripper_stop_speed,.5)
                original=io.gripper_status
                def status():
                    result=original()
                    result['motor_velocities']=[.1,0.]
                    return result
                io.gripper_status=status
                b.trip('fixed_target_test')
                for _ in range(15):
                    now[0]+=.02
                    b.poll()
                self.assertEqual(b.state,'HOLD_CONFIRMED')
                self.assertFalse(b.stop_report['stationary_feedback_confirmed'])
                self.assertTrue(b.stop_report['joint_stationary_feedback_confirmed'])
                self.assertFalse(b.stop_report['hardware_stop_verified'])
                b.release(token)
            finally:
                b.close()

    def test_override_keeps_gripper_displacement_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            io=FakeIO(directory,lambda:10.)
            b=MeasuredHoldBackend(io,np.tile([-2.,2.],(14,1)),np.ones(14),
                Path(directory)/'execution.json',clock=lambda:10.,
                gripper_monitor_limit_rad_s=.5)
            b.start_watchdog()
            try:
                b.acquire('test')
                original=io.gripper_status
                def status():
                    result=original()
                    result['actual']=[.52,.5]
                    return result
                io.gripper_status=status
                with self.assertRaisesRegex(ControlFault,'gripper_displacement'):
                    b.checked_feedback()
            finally:
                b.close()

    def test_override_requires_finite_bounded_value(self):
        for value in [0.,.6,float('nan'),True]:
            with tempfile.TemporaryDirectory() as directory:
                io=FakeIO(directory,lambda:10.)
                with self.assertRaises(ValueError):
                    MeasuredHoldBackend(io,np.tile([-2.,2.],(14,1)),np.ones(14),
                        Path(directory)/'execution.json',gripper_monitor_limit_rad_s=value)
