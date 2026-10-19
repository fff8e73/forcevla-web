import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

from robot_control.action_adapter import prepare_actions
from robot_control.common import ROOT, sha256
from robot_control.hardware_backend import audit_prediction, SimulatedBackend
from robot_control.command_bridge import CommandBridge, Prediction


class ActionAdapterTests(unittest.TestCase):
    def setUp(self):
        self.raw = np.zeros((1, 50, 14), np.float32)
        self.raw[:, :, 0] = 0.5
        self.raw[:, :, 7] = 0.5
        self.raw[:, :, 6] = -0.014
        self.raw[:, :, 13] = 1.012

    def test_clips_only_grippers_preserves_raw_and_tcp(self):
        original = self.raw.copy()
        candidate, report = prepare_actions(self.raw, 0.05)
        np.testing.assert_array_equal(self.raw, original)
        np.testing.assert_array_equal(
            candidate[:, :, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]],
            original[:, :, [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]],
        )
        self.assertEqual(report["changed_values"], 100)
        np.testing.assert_array_equal(candidate[:, :, 6], 0)
        np.testing.assert_array_equal(candidate[:, :, 13], 1)
        self.assertEqual(candidate.dtype, self.raw.dtype)

    def test_valid_values_unchanged_and_independent(self):
        self.raw[:, :, 6] = 0.6
        self.raw[:, :, 13] = 1
        candidate, report = prepare_actions(self.raw, 0.05)
        self.assertEqual(report["changed_values"], 0)
        np.testing.assert_array_equal(candidate, self.raw)
        candidate[0, 0, 0] = 1
        self.assertEqual(self.raw[0, 0, 0], 0.5)

    def test_large_correction_withholds_candidate(self):
        for value in [-0.2, 1.2, 100]:
            self.raw[0, 49, 13] = value
            candidate, report = prepare_actions(self.raw, 0.05)
            self.assertIsNone(candidate)
            self.assertEqual(report["reasons"], ["gripper_correction_excessive"])

    def test_nonfinite_any_column_rejected(self):
        for col in [0, 6, 13]:
            raw = self.raw.copy()
            raw[0, 0, col] = np.nan
            with self.assertRaises(ValueError):
                prepare_actions(raw, 0.05)

    def test_invalid_budget_and_shape(self):
        for value in [-1, float("nan"), True]:
            with self.assertRaises(ValueError):
                prepare_actions(self.raw, value)
        with self.assertRaises(ValueError):
            prepare_actions(self.raw[0], 0.05)

    def fixture(self, folder):
        candidate, adaptation = prepare_actions(self.raw, 0.05)
        np.savez(folder / "prediction.npz", actions=self.raw)
        np.savez(folder / "execution_candidate.npz", actions=candidate)
        (folder / "action_adaptation.json").write_text(json.dumps(adaptation))
        (folder / "ik_checks.json").write_text(
            json.dumps({"environment_collision_checked": False, "manifest": {}})
        )
        report = {
            "status": "accepted_by_diagnostic_gates",
            "artifacts": {p.name: sha256(p) for p in folder.iterdir() if p.is_file()},
            "snapshot": {"snapshot_unix_ns": 0},
            "action_reasons": [],
            "ik_reasons": [],
            "post_analysis_reasons": [],
        }
        (folder / "report.json").write_text(json.dumps(report))

    def test_audit_checks_candidate_against_raw_not_raw_range_only(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            self.fixture(folder)
            reasons = audit_prediction(folder)["reasons"]
            self.assertNotIn("gripper_out_of_range", reasons)
            self.assertNotIn("execution_candidate_mismatch", reasons)
            self.assertIn("saved_prediction_stale_not_replayable", reasons)
            candidate = np.load(folder / "execution_candidate.npz")["actions"]
            candidate[0, 0, 0] += 0.1
            np.savez(folder / "execution_candidate.npz", actions=candidate)
            reasons = audit_prediction(folder)["reasons"]
            self.assertIn("execution_candidate_mismatch", reasons)
            self.assertIn("artifact_hash_mismatch:execution_candidate.npz", reasons)

    def test_bridge_clips_at_execution_boundary_without_feedback_change(self):
        limits = json.loads((ROOT / "config/control_simulation.json").read_text())
        state = np.array(
            [
                0.5,
                0.1,
                0.7,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0.5,
                -0.1,
                0.7,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            float,
        )
        backend = SimulatedBackend(state)
        bridge = CommandBridge(backend, limits)
        bridge.connect(10)
        actions = np.tile(np.r_[state[:7], state[13:20]], (50, 1))
        actions[:, [6, 13]] = -0.014
        bridge.request("test", 10, 10)
        bridge.submit(Prediction("test", 10, actions, True), 10)
        bridge.tick(10)
        np.testing.assert_array_equal(actions[:, [6, 13]], -0.014)
        self.assertEqual(bridge.state, "EXECUTING")
        self.assertEqual(backend.write_count, 1)
        np.testing.assert_array_equal(bridge.target[[6, 13]], 0)
        self.assertTrue(any(e["event"] == "action_preparation" for e in bridge.events))
        bridge.close(10.005)


if __name__ == "__main__":
    unittest.main()
