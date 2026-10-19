"""No ROS construction, CAN access or real claim is used by these tests."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from robot_control.command_bridge import ControlFault
from robot_control.planner_loop import LoopPlannerIO
from robot_control.run_first10 import PlannerIO


class ClaimPolicyTests(unittest.TestCase):
    def test_clients_claim_with_large_tracking_error_and_velocity(self):
        for cls in (LoopPlannerIO, PlannerIO):
            calls = []
            fake = SimpleNamespace(
                require_motion=lambda: None, require_execution=lambda: None,
                gripper=lambda: dict(boot_id='boot', reserved=False,
                                     actual=[0.3, 0.3], applied_target=[0., 0.],
                                     motor_velocities=[0.1, -0.1]),
                grip_request=lambda action: calls.append(action) or {'ok': True})
            with self.subTest(cls=cls.__name__):
                self.assertEqual(cls.claim(fake), {'ok': True})
                self.assertEqual(calls, ['claim'])
                self.assertEqual(fake.boot, 'boot')

    def test_clients_keep_reservation_and_cancellation_guards(self):
        for cls in (LoopPlannerIO, PlannerIO):
            def reject():
                raise AssertionError('must not send a claim')
            fake = SimpleNamespace(
                require_motion=lambda: None, require_execution=lambda: None,
                gripper=lambda: dict(boot_id='boot', reserved=True),
                grip_request=lambda action: reject())
            with self.subTest(cls=cls.__name__):
                with self.assertRaises(ControlFault):
                    cls.claim(fake)
        def cancelled():
            raise ControlFault('motion_cancelled')
        fake.require_motion = cancelled
        with self.assertRaisesRegex(ControlFault, 'motion_cancelled'):
            LoopPlannerIO.claim(fake)

    def test_claim_rejection_is_reported_without_stationary_timeout(self):
        def reject(action):
            raise ControlFault('gripper_claim_rejected:unhealthy')
        for cls in (LoopPlannerIO, PlannerIO):
            fake = SimpleNamespace(
                require_motion=lambda: None, require_execution=lambda: None,
                gripper=lambda: dict(boot_id='boot', reserved=False),
                grip_request=reject)
            with self.assertRaisesRegex(ControlFault, 'gripper_claim_rejected:unhealthy'):
                cls.claim(fake)

    def test_driver_keeps_health_ownership_boot_and_range_guards(self):
        path = os.environ.get('FORCEVLA_GRIPPER_RULES_PATH',
            '/opt/kernelmind/apex_tool/install/lib/python3.12/site-packages/dm_gripper_py/web_gripper.py')
        spec = importlib.util.spec_from_file_location('claim_rules_under_test', path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for case in ('healthy', 'unhealthy', 'occupied', 'bad_boot', 'bad_target'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as root:
                gate = module.Gate(Path(root) / 'lease.json', clock=lambda: 10.)
                if case == 'occupied':
                    gate.reserved = True
                    gate.session = 'existing-owner'
                motors = [SimpleNamespace(getPosition=lambda: 0.49,
                    getVelocity=lambda: 0.1, getErr=lambda: 1 if case != 'unhealthy' else 10,
                    last_rx_monotonic=1.) for _ in range(2)]
                fake = SimpleNamespace(gate=gate,
                    node=SimpleNamespace(q1=2. if case == 'bad_target' else 0., q2=0.),
                    read_motors=lambda: [(m, 0., 1.6) for m in motors],
                    publish_status=lambda: None)
                fake.feedback = lambda: module.WebGripper.feedback(fake)
                fake.fault_reason = module.WebGripper.fault_reason
                fake.apply = lambda: module.WebGripper.apply(fake)
                fake.reserved = gate.reserved
                # apply() reads the live property in production; dummy is set
                # after Gate.command below by this wrapper.
                def apply():
                    fake.reserved = gate.reserved
                    module.WebGripper.apply(fake)
                fake.apply = apply
                req = dict(action='claim', boot_id='old-boot' if case == 'bad_boot' else gate.boot,
                           request_id='test-claim')
                before = (fake.node.q1, fake.node.q2)
                module.WebGripper.command(fake, SimpleNamespace(data=json.dumps(req)))
                self.assertEqual(gate.result['ok'], case == 'healthy')
                self.assertEqual((fake.node.q1, fake.node.q2), before)
                if case == 'healthy':
                    self.assertEqual(gate.target, [0., 0.])
                if case == 'occupied':
                    self.assertEqual(gate.session, 'existing-owner')


if __name__ == '__main__':
    unittest.main()
