from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_control.common import load_config
from robot_control.observation import (
    ObservationStore,
    Rejected,
    gripper_values,
    pose6,
    training_tiles,
    live_training_tiles,
    wrench6,
)
from robot_control.run_preflight import numeric_metrics, subset_differences
from robot_control.ros_observer import ROSObserver


class ObservationTests(unittest.TestCase):
    def setUp(self):
        self.config = load_config()
        self.store = ObservationStore(self.config)
        self.wall = 10_000_000_000
        self.mono = 20_000_000_000
        self.source = self.wall - 20_000_000

    def fill(self):
        data = {
            "left_pose": np.arange(6),
            "right_pose": np.arange(6) + 10,
            "left_wrench": np.arange(6) + 20,
            "right_wrench": np.arange(6) + 30,
            "gripper": (np.array([-0.01, 0.9]), np.array([0.001, 0.002])),
            "camera": {
                k: np.zeros((480, 640, 3), np.uint8)
                for k in ["image", "right_image", "wrist_image", "right_wrist_image"]
            },
        }
        for key, value in data.items():
            self.store.update(
                key, value, self.source, receipt_ns=self.wall, receipt_mono_ns=self.mono
            )
        return data

    def snapshot(self, advance=0):
        return self.store.snapshot(
            now_ns=self.wall + advance, now_mono_ns=self.mono + advance
        )

    def test_state_layout_and_no_gripper_clipping(self):
        self.fill()
        request, metadata = self.snapshot()
        np.testing.assert_allclose(
            request["state"],
            [
                0,
                1,
                2,
                3,
                4,
                5,
                -0.01,
                20,
                21,
                22,
                23,
                24,
                25,
                10,
                11,
                12,
                13,
                14,
                15,
                0.9,
                30,
                31,
                32,
                33,
                34,
                35,
            ],
        )
        self.assertEqual(request["state"].dtype, np.float32)
        self.assertFalse(metadata["camera_mapping_verified"])
        with self.assertRaisesRegex(Rejected, "no_new_camera"):
            self.snapshot()

    def test_missing_never_filled(self):
        with self.assertRaisesRegex(Rejected, "missing"):
            self.snapshot()

    def test_duplicate_does_not_refresh_receipt(self):
        data = self.fill()
        with self.assertRaisesRegex(Rejected, "duplicate"):
            self.store.update(
                "left_pose",
                data["left_pose"],
                self.source,
                receipt_ns=self.wall + 100_000_000,
                receipt_mono_ns=self.mono + 100_000_000,
            )
        with self.assertRaisesRegex(Rejected, "stale"):
            self.snapshot(600_000_000)

    def test_future_and_old_at_receipt(self):
        for source, reason in [
            (self.wall + 50_000_000, "future"),
            (self.wall - 1_000_000_000, "stale"),
        ]:
            with self.assertRaisesRegex(Rejected, reason):
                self.store.update(
                    "left_pose",
                    np.zeros(6),
                    source,
                    receipt_ns=self.wall,
                    receipt_mono_ns=self.mono,
                )

    def test_causal_alignment_ignores_pose_after_image(self):
        self.fill()
        self.store.update(
            "left_pose",
            np.ones(6) * 999,
            self.wall,
            receipt_ns=self.wall,
            receipt_mono_ns=self.mono,
        )
        request, _ = self.snapshot()
        np.testing.assert_array_equal(request["state"][:6], np.arange(6))

    def test_skew_rejected(self):
        data = self.fill()
        with self.store.lock:
            self.store.histories["left_pose"].clear()
        self.store.update(
            "left_pose",
            data["left_pose"],
            self.source - 200_000_000,
            receipt_ns=self.wall,
            receipt_mono_ns=self.mono,
        )
        with self.assertRaisesRegex(Rejected, "skew"):
            self.snapshot()

    def test_underlying_motor_age_includes_elapsed_time(self):
        self.fill()
        with self.assertRaisesRegex(Rejected, "stale_motor_feedback"):
            self.snapshot(190_000_000)

    def test_invalid_stream_latched_until_valid_update(self):
        data = self.fill()
        self.store.reject("left_pose", "invalid quaternion")
        with self.assertRaisesRegex(Rejected, "invalid_stream"):
            self.snapshot()
        self.store.update(
            "left_pose",
            data["left_pose"],
            self.source + 1,
            receipt_ns=self.wall,
            receipt_mono_ns=self.mono,
        )
        # Camera must also advance to allow the recovered causal sample.
        self.store.update(
            "camera",
            data["camera"],
            self.source + 2,
            receipt_ns=self.wall,
            receipt_mono_ns=self.mono,
        )
        self.snapshot()

    def test_quaternion_convention_and_bad_frame(self):
        q = Rotation.from_euler("xyz", [0.4, -0.2, 1.1]).as_quat()
        p, singular = pose6([0.1, 0.2, 0.3], q, "base_link")
        np.testing.assert_allclose(p, [0.1, 0.2, 0.3, 0.4, -0.2, 1.1], atol=1e-12)
        self.assertFalse(singular)
        with self.assertRaisesRegex(Rejected, "frame"):
            pose6([0, 0, 0], q, "world")
        with self.assertRaisesRegex(Rejected, "nonunit"):
            pose6([0, 0, 0], [0, 0, 0, 0], "base_link")
        with self.assertRaises(Rejected):
            pose6([np.nan, 0, 0], q, "base_link")

    def test_wrench_keeps_original_components(self):
        np.testing.assert_array_equal(
            wrench6([1, 2, 3], [4, 5, 6], "flange_L", "flange_L"), [1, 2, 3, 4, 5, 6]
        )
        with self.assertRaises(Rejected):
            wrench6([1, 2, 3], [4, 5, 6], "base_link", "flange_L")

    def test_gripper_actual_not_divided_twice(self):
        obj = {
            "healthy": True,
            "actual": [-0.02, 0.5],
            "calibration": [[0, 1.6], [0, 1.6]],
            "motor_feedback_age": [0.01, 0.02],
        }
        actual, _ = gripper_values(obj, self.config["expected_calibration"])
        np.testing.assert_array_equal(actual, [-0.02, 0.5])
        obj["calibration"][0][1] = 1.7
        with self.assertRaisesRegex(Rejected, "calibration"):
            gripper_values(obj, self.config["expected_calibration"])

    def test_camera_order_rgb_and_padding(self):
        rgb = np.zeros((1080, 1920, 3), np.uint8)
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (200, 200, 200)]
        for color, (x, y) in zip(colors, [(0, 0), (1, 0), (0, 1), (1, 1)]):
            rgb[y * 540 : (y + 1) * 540, x * 960 : (x + 1) * 960] = color
        tiles = training_tiles(rgb)
        for image, color in zip(tiles.values(), colors):
            self.assertEqual(image.shape, (480, 640, 3))
            np.testing.assert_allclose(image[240, 320], color, atol=2)
            self.assertLess(int(image[10].max()), 3)
            self.assertLess(int(image[470].max()), 3)
        with self.assertRaises(Rejected):
            training_tiles(rgb[:100])

    def test_live_geometry_is_explicit_and_undeclared_size_rejected(self):
        rgb = np.zeros((1488, 1920, 3), np.uint8)
        rgb[:744, :960] = [255, 0, 0]
        tiles = live_training_tiles(rgb, [1488, 1920, 3])
        np.testing.assert_allclose(tiles["image"][240, 320], [255, 0, 0], atol=2)
        with self.assertRaises(Rejected):
            live_training_tiles(rgb, [1080, 1920, 3])

    def test_decoded_frame_backlog_keeps_latest_without_resetting_decoder(self):
        observer = ROSObserver(self.config, self.store)
        for i in range(10):
            observer.queue_frame(
                object(), self.source + i, self.wall, self.mono, "h264"
            )
        self.assertEqual(observer.frames.qsize(), 1)
        self.assertEqual(observer.frames.get_nowait()[1], self.source + 9)
        self.assertFalse(observer.reset_decoder.is_set())

    def test_runtime_contract_changes_report_path(self):
        errors = subset_differences(
            {"camera": {"fps": 30}}, {"camera": {"fps": 20, "other": 1}}, "quad"
        )
        self.assertIn("quad.camera.fps", errors[0])
        self.assertEqual(subset_differences({"fps": 30}, {"fps": 30, "other": 1}), [])

    def test_original_numerical_gates_not_only_shape(self):
        t = {
            "rmse_max": 0.02,
            "absolute_error_max": 0.15,
            "relative_l2_max": 0.02,
            "relative_l2_denominator_floor": 1e-6,
        }
        expected = np.ones((1, 50, 32), np.float32)
        self.assertTrue(numeric_metrics(expected.copy(), expected, t)["passed"])
        actual = expected.copy()
        actual[0, 0, 0] += 0.2
        self.assertFalse(numeric_metrics(actual, expected, t)["passed"])
        actual = expected.copy()
        actual[0, 0, 31] = np.nan
        with self.assertRaises(ValueError):
            numeric_metrics(actual, expected, t)


if __name__ == "__main__":
    unittest.main()
