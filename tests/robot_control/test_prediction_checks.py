import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

from robot_control.common import ROOT, load_config
from robot_control.observation import ObservationStore
from robot_control.offline_ik import OfflineIK, JOINTS, pose_delta
from robot_control.prediction_checks import check_actions


class PredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits = json.loads((ROOT / "config/planning.json").read_text())
        cls.solver = OfflineIK(cls.limits["expected_qp_parameters"], cls.limits)
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
        cls.poses = cls.solver.poses(cls.q)
        cls.state = np.r_[
            cls.poses[0], 0.5, np.zeros(6), cls.poses[1], 0.5, np.zeros(6)
        ]
        cls.actions = np.tile(np.r_[cls.poses[0], 0.5, cls.poses[1], 0.5], (50, 1))

    def test_rpc_observation_id_mismatch_and_invalid_output(self):
        sys.path.insert(0, load_config()["repository"])
        from robot_control.model_rpc import predict_observation
        from examples.forcevla.transport import model_inference_pb2 as pb
        from examples.forcevla.transport.codec import encode
        from types import SimpleNamespace

        request = {
            k: np.zeros((4, 4, 3), np.uint8)
            for k in ["image", "right_image", "wrist_image", "right_wrist_image"]
        }
        request.update(state=np.zeros(26, np.float32), prompt="test", seed=0)
        response = pb.InferenceResponse(request_id="wrong")
        policy = SimpleNamespace(
            model_name="forcevla-dual",
            timeout=1,
            stub=SimpleNamespace(Infer=lambda *a, **kw: response),
        )
        with self.assertRaisesRegex(RuntimeError, "observation ID"):
            predict_observation(policy, request, "expected")
        response.request_id = "expected"
        with self.assertRaisesRegex(ValueError, "prediction"):
            predict_observation(policy, request, "expected")
        response.outputs["actions"].CopyFrom(encode(np.zeros((1, 50, 14), np.float32)))
        response.outputs["normalized_actions"].CopyFrom(
            encode(np.zeros((1, 50, 32), np.float32))
        )
        self.assertEqual(
            predict_observation(policy, request, "expected")["actions"].shape,
            (1, 50, 14),
        )

    def test_hold_pose_full_horizon(self):
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(self.actions, self.q, self.state, path)
            self.assertEqual(result["status"], "passed_scoped_ik", result["reasons"])
            self.assertEqual(len(result["targets"]), 50)
            self.assertFalse(result["environment_collision_checked"])
            self.assertFalse(result["manifest"]["online_qp_equivalence_verified"])
            q = np.load(Path(path) / "ik_trajectory.npz")["joint_positions"]
            self.assertLess(float(np.max(np.abs(q - self.q))), 1e-5)

    def test_small_reachable_motion(self):
        actions = self.actions.copy()
        actions[:, 0] += np.linspace(0, 0.002, 50)
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(actions, self.q, self.state, path)
            self.assertEqual(len(result["targets"]), 50)
            self.assertNotIn("ik_residual", result["reasons"])
            self.assertLess(result["targets"][-1]["position_error_m"][0], 0.001)

    def test_bad_seed_fk_is_rejected(self):
        state = self.state.copy()
        state[0] += 0.1
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(self.actions, self.q, state, path)
            self.assertIn("seed_fk_observation_mismatch", result["reasons"])

    def test_outside_joint_bounds_seed_rejected(self):
        q = self.q.copy()
        q[0] = 5
        pose = self.solver.poses(q)
        state = np.r_[pose[0], 0.5, np.zeros(6), pose[1], 0.5, np.zeros(6)]
        with tempfile.TemporaryDirectory() as path:
            result = self.solver.rollout(self.actions, q, state, path)
            self.assertIn("seed_joint_limit", result["reasons"])

    def test_unreachable_target_and_solver_budget(self):
        limits = dict(self.limits, max_nfev=4, solver_timeout_s=4)
        solver = OfflineIK(limits["expected_qp_parameters"], limits)
        actions = self.actions.copy()
        actions[:, 0] = 20
        with tempfile.TemporaryDirectory() as path:
            result = solver.rollout(actions, self.q, self.state, path)
            self.assertEqual(result["status"], "rejected")
            self.assertIn("ik_residual", result["reasons"])

    def test_post_analysis_age_not_renewed_by_fast_rpc(self):
        from robot_control.prediction_checks import check_analysis_age

        self.assertEqual(check_analysis_age(1.6, 2.5), [])
        self.assertEqual(check_analysis_age(2.6, 2.5), ["analysis_result_stale"])
        with self.assertRaises(ValueError):
            check_analysis_age(float("nan"), 2.5)

    def test_response_age_gripper_and_pose_rejections(self):
        actions = self.actions[None].copy()
        actions[0, 1, 6] = 1.001
        actions[0, 4, 0] += 1
        result = check_actions(actions, self.state, 3, self.limits)
        for key in [
            "response_stale",
            "gripper_out_of_range",
            "tcp_displacement",
            "tcp_step_translation",
        ]:
            self.assertIn(key, result["reasons"])
        self.assertEqual(actions[0, 1, 6], 1.001)

    def test_rotation_wrap_is_small_geodesic(self):
        left = np.array([0, 0, 0, 0, 0, np.pi - 0.001])
        right = np.array([0, 0, 0, 0, 0, -np.pi + 0.001])
        self.assertAlmostEqual(pose_delta(left, right)[1], 0.002, places=6)

    def test_collision_pairs_all_resolve(self):
        distances = self.solver.collisions(self.q)
        self.assertEqual(len(distances), 8)
        self.assertTrue(all(np.isfinite(x["distance_m"]) for x in distances))
        # Explicitly tightened margin rejects this known pose, exercising the gate.
        parameters = dict(
            self.limits["expected_qp_parameters"], collision_minimum_distance=5.0
        )
        solver = OfflineIK(parameters, self.limits)
        with tempfile.TemporaryDirectory() as path:
            self.assertIn(
                "seed_collision_margin",
                solver.rollout(self.actions, self.q, self.state, path)["reasons"],
            )

    def test_joints_required_only_for_prediction_and_name_mapping(self):
        config = dict(load_config(), require_joints=True)
        store = ObservationStore(config)
        self.assertIn("joints", store.histories)
        from robot_control.ros_observer import ROSObserver
        from types import SimpleNamespace

        observer = ROSObserver(config, store)
        import time

        now = time.time_ns()
        message = SimpleNamespace(
            name=JOINTS[::-1],
            position=self.q[::-1].tolist(),
            header=SimpleNamespace(
                stamp=SimpleNamespace(sec=now // 10**9, nanosec=now % 10**9)
            ),
        )
        observer.numeric_callback("joints")(message)
        np.testing.assert_array_equal(store.histories["joints"][-1].value, self.q)
        message.name[0] = message.name[1]
        observer.numeric_callback("joints")(message)
        self.assertIn("joints", store.errors)


if __name__ == "__main__":
    unittest.main()
