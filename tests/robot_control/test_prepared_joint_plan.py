import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.common import ROOT
from robot_control.native_qp import NativeQP
from robot_control.prepared_joint_plan import prepare_native_plan


class PlanTests(unittest.TestCase):
    def test_installed_core_to_joint_envelopes_preserves_time_and_prefix(self):
        limits = json.loads((ROOT / "config/planning.json").read_text())
        solver = NativeQP(limits["expected_qp_parameters"], limits)
        q = np.array(
            [
                0.69,
                -1.36,
                -0.75,
                -0.90,
                -0.23,
                0.24,
                -1.05,
                -1.29,
                -1.09,
                1.14,
                -1.26,
                0.58,
                -0.19,
                -0.035,
            ]
        )
        poses = solver.poses(q)
        state = np.r_[poses[0], 0.5, np.zeros(6), poses[1], 0.5, np.zeros(6)]
        actions = np.tile(np.r_[poses[0], 0.5, poses[1], 0.5], (1, 50, 1))
        with tempfile.TemporaryDirectory() as folder:
            plan = prepare_native_plan(
                solver,
                actions=actions,
                seed=q,
                state=state,
                request_id="synthetic_native",
                observed_at=10.0,
                output=Path(folder) / "plan",
                prefix_steps=3,
                max_correction=0.05,
                clock=lambda: 10.1,
            )
        self.assertEqual(plan.joints.shape, (50, 14))
        self.assertEqual(plan.grippers.shape, (50, 2))
        self.assertEqual(plan.dt, 0.002)
        self.assertFalse(plan.joints.flags.writeable)
        envelope = plan.envelope("token", 49, 52, 10.2)
        self.assertEqual(envelope.observation_time, 10.0)
        self.assertEqual(envelope.sequence, 52)
        self.assertFalse(plan.manifest["hardware_execution_ready"])
        self.assertLess(np.max(np.abs(plan.joints - q)), 1e-8)

    def test_old_observation_rejected_before_solver(self):
        solver = Mock()
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ControlFault, "stale"),
        ):
            prepare_native_plan(
                solver,
                actions=np.zeros((1, 50, 14)),
                seed=np.zeros(14),
                state=np.zeros(26),
                request_id="old",
                observed_at=1.0,
                output=Path(folder) / "plan",
                prefix_steps=1,
                max_correction=0.05,
                clock=lambda: 10.0,
            )
        solver.rollout.assert_not_called()

    def test_no_refresh_after_slow_solver(self):
        solver = Mock()
        solver.rollout.return_value = {
            "status": "passed_scoped_qp",
            "reasons": [],
            "manifest": {"same_installed_qp_core": True},
        }
        clock = Mock(side_effect=[10.0, 14.0])
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ControlFault, "expired_during_qp"),
        ):
            prepare_native_plan(
                solver,
                actions=np.zeros((1, 50, 14)),
                seed=np.zeros(14),
                state=np.zeros(26),
                request_id="slow",
                observed_at=10.0,
                output=Path(folder) / "plan",
                prefix_steps=1,
                max_correction=0.05,
                clock=clock,
            )

    def test_rejected_qp_never_prepares_prefix(self):
        solver = Mock()
        solver.rollout.return_value = {"status": "rejected", "reasons": ["collision"]}
        with (
            tempfile.TemporaryDirectory() as folder,
            self.assertRaisesRegex(ControlFault, "qp_checks_failed"),
        ):
            prepare_native_plan(
                solver,
                actions=np.zeros((1, 50, 14)),
                seed=np.zeros(14),
                state=np.zeros(26),
                request_id="rejected",
                observed_at=10.0,
                output=Path(folder) / "plan",
                prefix_steps=1,
                max_correction=0.05,
                clock=lambda: 10.1,
            )
