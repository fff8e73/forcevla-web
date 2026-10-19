import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from robot_control.common import ROOT
from robot_control.native_qp import NativeQP


class NativeQPTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = json.loads((ROOT / "config/planning.json").read_text())
        cls.solver = NativeQP(cls.limits["expected_qp_parameters"], cls.limits)
        cls.q = np.array(
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
        pose = cls.solver.poses(cls.q)
        cls.state = np.r_[pose[0], 0.5, np.zeros(6), pose[1], 0.5, np.zeros(6)]
        cls.actions = np.tile(np.r_[pose[0], 0.5, pose[1], 0.5], (50, 1))

    def test_hold_834_ticks_and_repeatability(self):
        trajectories = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as path:
                result = self.solver.rollout(self.actions, self.q, self.state, path)
                self.assertEqual(
                    result["status"], "passed_scoped_qp", result["reasons"]
                )
                self.assertEqual(result["simulated_ticks"], 834)
                self.assertTrue(result["manifest"]["same_installed_qp_core"])
                self.assertFalse(result["manifest"]["online_qp_equivalence_verified"])
                d = np.load(Path(path) / "qp_trajectory.npz")
                trajectories.append(d["joint_positions"].copy())
                self.assertLess(np.max(abs(trajectories[-1] - self.q)), 1e-8)
        np.testing.assert_array_equal(*trajectories)

    def test_small_motion_time_tracking(self):
        actions = self.actions.copy()
        actions[:, 0] += np.linspace(0, 0.003, 50)
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(actions, self.q, self.state, path)
            self.assertEqual(result["status"], "passed_scoped_qp", result["reasons"])
            self.assertLess(result["targets"][-1]["position_error_m"][0], 0.001)
            self.assertEqual(len(result["targets"]), 50)

    def test_far_target_not_marked_executable(self):
        actions = self.actions.copy()
        actions[:, 0] += 10
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(actions, self.q, self.state, path)
            self.assertEqual(result["status"], "rejected")
            self.assertTrue(
                set(result["reasons"]) & {"finite_time_tracking_error", "qp_failure"}
            )
            self.assertFalse(result["hardware_execution_ready"])

    def test_invalid_seed_rejected_before_native(self):
        state = self.state.copy()
        state[0] += 0.2
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(self.actions, self.q, state, path)
            self.assertIn("seed_fk_observation_mismatch", result["reasons"])
            self.assertFalse((Path(path) / "qp_input.json").exists())

    def test_native_timeout_recorded(self):
        limits = dict(self.limits, solver_timeout_s=0.000001)
        solver = NativeQP(limits["expected_qp_parameters"], limits)
        with tempfile.TemporaryDirectory() as path:
            result = solver.rollout(self.actions, self.q, self.state, path)
            self.assertEqual(result["reasons"], ["solver_timeout"])

    def test_nonfinite_rejected(self):
        actions = self.actions.copy()
        actions[0, 0] = np.nan
        with tempfile.TemporaryDirectory() as path:
            with self.assertRaises(ValueError):
                self.solver.rollout(actions, self.q, self.state, path)


if __name__ == "__main__":
    unittest.main()
