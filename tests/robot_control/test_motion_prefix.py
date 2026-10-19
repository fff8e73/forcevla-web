import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import ROOT
from robot_control.motion_prefix import check_motion_prefix, PrefixScheduler, hardware_trial_blockers
from robot_control.prepared_joint_plan import PreparedJointPlan
from robot_control.scheduled_joint_backend import ScheduledJointBackend
from test_joint_execution import FakeIO


class Solver:
    def poses(self, q):
        return np.array([np.r_[q[:3], q[3:6]], np.r_[q[7:10], q[10:13]]])


class SchedulerBackend:
    def __init__(self, clock, advance):
        self.io = SimpleNamespace(production=False)
        self.clock = clock
        self.advance = advance
        self.sent = []
        self.holds = []
        self.fail_at = None
        self.block_s = 0
        self.grippers = []
        self.scheduler_capabilities = {
            "bounded_joint_submit": True,
            "independent_gripper_lane": True,
        }

    def checked_feedback(self, **kwargs):
        return SimpleNamespace(joints=np.zeros(14))

    def heartbeat(self, token):
        pass

    def queue_gripper(self, *args):
        self.grippers.append(args)

    def write_joint_tick(self, e):
        if len(self.sent) == self.fail_at:
            raise ConnectionError("injected partial submission")
        self.sent.append(e)
        self.advance(self.block_s)

    def trip(self, reason):
        self.holds.append(reason)


class PrefixTests(unittest.TestCase):
    def setUp(self):
        self.profile = json.loads((ROOT / "config/stage_d_trial.json").read_text())
        self.now = 10.1
        self.q = np.zeros((17, 14))
        self.q[:, 0] = 1e-5 * (1 - np.cos(np.linspace(np.pi / 17, np.pi, 17))) / 2
        self.plan = PreparedJointPlan(
            "synthetic",
            10.0,
            np.zeros(14),
            self.q,
            np.ones((17, 2)) * 0.5,
            0.002,
            {"prefix_steps": 1},
        )

    def advance(self, d):
        self.now += d

    def check(self, plan=None):
        return check_motion_prefix(
            plan or self.plan,
            Solver(),
            self.profile,
            seed_velocity=np.zeros(14),
            initial_gripper_targets=np.array([0.5, 0.5]),
        )

    def test_smooth_tiny_prefix_includes_stop_boundary(self):
        check = self.check()
        self.assertEqual(check["status"], "passed_prefix_budgets")
        self.assertTrue(check["terminal_target_hold_checked"])

    def test_joint_and_tcp_displacement_rejected(self):
        self.q[:, 0] = 0.1
        check = self.check()
        self.assertIn("trial_joint_displacement", check["reasons"])
        self.assertIn("trial_tcp_displacement", check["reasons"])

    def test_abrupt_stop_acceleration_rejected(self):
        self.q[:, 0] = np.arange(1, 18) * 0.0001
        check = self.check()
        self.assertIn("trial_acceleration_or_stop_boundary", check["reasons"])

    def test_gripper_change_is_not_silently_ignored(self):
        self.plan.grippers[:, 0] = 0.51
        check = self.check()
        self.assertIn("trial_gripper_speed", check["reasons"])

    def test_scheduler_preserves_original_clock_and_prefix(self):
        backend = SchedulerBackend(lambda: self.now, self.advance)
        scheduler = PrefixScheduler(
            backend, self.profile, clock=lambda: self.now, sleep=self.advance
        )
        result = scheduler.execute(self.plan, "token", self.check())
        self.assertEqual(result["status"], "prefix_submitted_hold_requested")
        self.assertEqual(len(backend.sent), 17)
        self.assertTrue(all(e.observation_time == 10.0 for e in backend.sent))
        self.assertEqual(backend.holds, ["finite_prefix_complete"])
        self.assertEqual(len(backend.grippers), 1)

    def test_no_catchup_after_blocking_submission(self):
        backend = SchedulerBackend(lambda: self.now, self.advance)
        backend.block_s = 0.02
        result = PrefixScheduler(
            backend, self.profile, clock=lambda: self.now, sleep=self.advance
        ).execute(self.plan, "token", self.check())
        self.assertEqual(result["status"], "failed")
        self.assertEqual(len(backend.sent), 1)
        self.assertIn("submission_deadline", result["error"])

    def test_cancel_and_failure_request_hold(self):
        for fail in [False, True]:
            with self.subTest(fail=fail):
                backend = SchedulerBackend(lambda: self.now, self.advance)
                if fail:
                    backend.fail_at = 0
                result = PrefixScheduler(
                    backend, self.profile, clock=lambda: self.now, sleep=self.advance
                ).execute(self.plan, "token", self.check(), cancel=lambda: not fail)
                self.assertEqual(result["status"], "failed")
                self.assertEqual(len(backend.holds), 1)

    def test_live_gate_cannot_be_bypassed_by_stage_A(self):
        reasons = hardware_trial_blockers(
            {"status": "passed_stationary_stage_A"}, self.profile
        )
        self.assertIn("stage_B_live_hold_acceptance_missing", reasons)
        backend = SchedulerBackend(lambda: self.now, self.advance)
        backend.io.production = True
        with self.assertRaisesRegex(ControlFault, "stage_B"):
            PrefixScheduler(backend, self.profile).execute(
                self.plan, "token", self.check()
            )
        self.assertEqual(backend.sent, [])

    def test_plan_tamper_and_inflight_gripper_rejected(self):
        backend = SchedulerBackend(lambda: self.now, self.advance)
        check = self.check()
        self.q[0, 0] = 0.1
        with self.assertRaisesRegex(ControlFault, "changed_after_validation"):
            PrefixScheduler(backend, self.profile, clock=lambda: self.now).execute(
                self.plan, "token", check
            )
        self.q[0, 0] = 0.0
        backend.gripper_pending = True
        result = PrefixScheduler(
            backend, self.profile, clock=lambda: self.now, sleep=self.advance
        ).execute(self.plan, "token", self.check())
        self.assertEqual(result["status"], "failed")
        self.assertIn("gripper_ack", result["error"])

    def test_old_plan_and_sync_backend_rejected(self):
        backend = SchedulerBackend(lambda: self.now, self.advance)
        self.now = 13.0
        with self.assertRaisesRegex(ControlFault, "expire"):
            PrefixScheduler(
                backend, self.profile, clock=lambda: self.now, sleep=self.advance
            ).execute(self.plan, "token", self.check())
        self.now = 10.1
        backend.scheduler_capabilities = {}
        with self.assertRaisesRegex(ControlFault, "synchronous"):
            PrefixScheduler(backend, self.profile, clock=lambda: self.now).execute(
                self.plan, "token", self.check()
            )

    def test_scheduled_adapter_uses_stage_B_and_keeps_audit(self):
        with tempfile.TemporaryDirectory() as folder:
            io = FakeIO(folder, lambda: self.now)
            backend = ScheduledJointBackend(
                io,
                np.tile([-2, 2], (14, 1)),
                np.ones(14),
                Path(folder) / "execution.json",
                clock=lambda: self.now,
            )
            try:
                token = backend.acquire("sandbox")
                scheduler = PrefixScheduler(
                    backend, self.profile, clock=lambda: self.now, sleep=self.advance
                )
                result = scheduler.execute(self.plan, token, self.check())
                self.assertEqual(result["status"], "prefix_submitted_hold_requested")
                for _ in range(20):
                    self.advance(0.02)
                    backend.poll()
                self.assertEqual(backend.state, "HOLD_CONFIRMED")
                backend.release(token)
                self.assertEqual(backend.state, "RELEASED")
            finally:
                backend.close()
            lines = [json.loads(s) for s in backend.audit_path.read_text().splitlines()]
            self.assertEqual(
                sum(x["event"] == "transport_submission" for x in lines), 17
            )

    def test_production_scheduled_adapter_not_enabled(self):
        with self.assertRaisesRegex(ControlFault, "live_scheduled"):
            ScheduledJointBackend(
                SimpleNamespace(production=True), None, None, "unused"
            )


if __name__ == "__main__":
    unittest.main()


class BFixedHoldEvidenceTests(unittest.TestCase):
    def test_real_fixed_hold_only_removes_B_prerequisite(self):
        profile=json.loads((ROOT / "config/stage_d_trial.json").read_text())
        report={"status":"passed_measured_joint_fixed_hold","live_requested":True,
                "final_state":"RELEASED","submitted_pairs":67,"production_publications":189,
                "stop_report":{"hold_feedback_confirmed":True},
                "final_feedback":{"mode":1,"input":0}}
        reasons=hardware_trial_blockers(report,profile)
        self.assertNotIn("stage_B_live_hold_acceptance_missing",reasons)
        self.assertIn("independent_stop_path_not_verified",reasons)
        report["live_requested"]=False
        self.assertIn("stage_B_live_hold_acceptance_missing",hardware_trial_blockers(report,profile))


class DiagnosticPrefixTests(unittest.TestCase):
    def test_five_step_diagnostic_does_not_unlock_executor(self):
        from robot_control.motion_prefix import validate_trial_profile
        profile=json.loads((ROOT / "config/stage_d_trial.json").read_text())
        profile["prefix_steps"]=5
        validate_trial_profile(profile,diagnostic_only=True)
        with self.assertRaises(ValueError):
            validate_trial_profile(profile)
        plan=PreparedJointPlan("diagnostic-five",10.,np.zeros(14),np.zeros((84,14)),
                               np.full((84,2),.5),.002,{"prefix_steps":5})
        check=check_motion_prefix(plan,Solver(),profile,seed_velocity=np.zeros(14),
                                  initial_gripper_targets=np.array([.5,.5]),diagnostic_only=True)
        self.assertEqual(check["status"],"passed_prefix_budgets")
        self.assertTrue(check["diagnostic_only"])
        self.assertFalse(check["hardware_execution_ready"])
        backend=SchedulerBackend(lambda:10.1,lambda _:None)
        with self.assertRaises(ValueError):
            PrefixScheduler(backend,profile)
        self.assertEqual(backend.sent,[])

    def test_even_short_diagnostic_cannot_execute(self):
        profile=json.loads((ROOT / "config/stage_d_trial.json").read_text())
        plan=PreparedJointPlan("diagnostic-one",10.,np.zeros(14),np.zeros((17,14)),
                               np.full((17,2),.5),.002,{"prefix_steps":1})
        check=check_motion_prefix(plan,Solver(),profile,seed_velocity=np.zeros(14),
                                  initial_gripper_targets=np.array([.5,.5]),diagnostic_only=True)
        backend=SchedulerBackend(lambda:10.1,lambda _:None)
        with self.assertRaisesRegex(ControlFault,"diagnostic_plan_cannot_execute"):
            PrefixScheduler(backend,profile,clock=lambda:10.1).execute(plan,'token',check)
        self.assertEqual(backend.sent,[])
