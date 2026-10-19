import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from robot_control.closed_loop_contracts import Barrier, CycleIdentity, LoopConfig, Observation
from robot_control.closed_loop_observation import (
    CAMERAS,
    JOINT_NAMES,
    SOURCES,
    ObservationPending,
    validated_observation,
)
from robot_control.closed_loop_planning import (
    PrefixPreparer,
    brake_path,
    constrained_prefix,
    kinematic_checks,
)
from robot_control.command_bridge import ControlFault
from robot_control.native_qp import NativeQP


class ContractsTests(unittest.TestCase):
    def test_invalid_configs(self):
        for change in (
            {"max_cycles": True},
            {"max_cycles": 0},
            {"execution_horizon": 50},
            {"max_prediction_age_s": float("nan")},
            {"control_period_s": 0.01},
            {"max_lateness_s": 0.003},
            {"gripper_stop_speed_rad_s": 0.5},
            {"unknown": 1},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                LoopConfig.from_dict(change)

    def observation(self):
        meta = dict(
            snapshot_id="boot-2",
            errors=[],
            sources={k: dict(source_time=100.1, received_mono=10.1) for k in SOURCES},
            joints=dict(
                names=JOINT_NAMES,
                positions=[0.0] * 14,
                source=dict(source_time=100.1, received_mono=10.1),
            ),
            control_feedback=dict(
                joints=dict(
                    positions=[0.0] * 14,
                    velocities=[0.0] * 14,
                    source_time=100.1,
                    received_mono=10.1,
                ),
                left_gripper=dict(
                    position_rad=0.5, velocity_rad_s=0.0, received_mono=10.1
                ),
                right_gripper=dict(
                    position_rad=0.5, velocity_rad_s=0.0, received_mono=10.1
                ),
            ),
            gripper_target_source=dict(source_time=100.1, received_mono=10.1),
            gripper_applied_target=[0.5, 0.5],
            motor_feedback_age=[0.01, 0.01],
        )
        images = {k: np.zeros((480, 640, 3), np.uint8) for k in CAMERAS}
        return meta, images

    def test_source_not_receipt_controls_newness(self):
        meta, images = self.observation()
        obs = validated_observation(
            meta,
            np.zeros(26),
            images,
            Barrier(10.0, 100.0),
            None,
            LoopConfig(),
            now=10.2,
            wall=100.2,
        )
        self.assertAlmostEqual(obs.observed_at, 10.09)
        self.assertFalse(obs.joints.flags.writeable)
        meta["sources"]["image"]["source_time"] = 99.9
        with self.assertRaisesRegex(ObservationPending, "before_hold"):
            validated_observation(
                meta,
                np.zeros(26),
                images,
                Barrier(10.0, 100.0),
                None,
                LoopConfig(),
                now=10.2,
                wall=100.2,
            )

    def test_motor_age_and_clock_jump(self):
        meta, images = self.observation()
        meta["motor_feedback_age"] = [0.2, 0.01]
        with self.assertRaisesRegex(ObservationPending, "motor_before_hold"):
            validated_observation(
                meta,
                np.zeros(26),
                images,
                Barrier(10.0, 100.0),
                None,
                LoopConfig(),
                now=10.2,
                wall=100.2,
            )
        with self.assertRaisesRegex(ControlFault, "clock_changed"):
            validated_observation(
                meta,
                np.zeros(26),
                images,
                Barrier(10.0, 100.0),
                None,
                LoopConfig(),
                now=10.2,
                wall=101.2,
            )


class ConstrainedPlanningTests(unittest.TestCase):
    def test_hard_derivative_bounds_and_terminal_rest(self):
        c = LoopConfig()
        seed, v0 = np.zeros(14), np.zeros(14)
        reference = np.tile(np.linspace(0.001, 0.008, 50)[:, None], (1, 14))
        bounds = np.tile([-2.0, 2.0], (14, 1))
        q = constrained_prefix(reference, seed, v0, bounds, c)
        failures, report, v = kinematic_checks(q, seed, v0, c, bounds)
        self.assertEqual(failures, [], report)
        self.assertGreater(q[-1, 0], 0)
        np.testing.assert_allclose(v[-1], 0, atol=1e-10)
        # Every mid-segment cancellation has an explicitly checked braking path.
        for row, velocity in zip(q, v):
            braking = brake_path(
                row, velocity, c.control_period_s, c.max_joint_acceleration_rad_s2
            )
            failures, _, _ = kinematic_checks(braking, row, velocity, c, bounds)
            self.assertEqual(failures, [])

    def test_expired_prepare(self):
        with self.assertRaisesRegex(ControlFault, "prepare_timeout"):
            constrained_prefix(
                np.zeros((17, 14)),
                np.zeros(14),
                np.zeros(14),
                np.tile([-2, 2], (14, 1)),
                LoopConfig(),
                deadline=0.0,
            )

    def test_measured_tracking_error_does_not_create_a_command_step(self):
        from robot_control.closed_loop_simulation import SyntheticGeometry, SyntheticPolicy

        solver = SyntheticGeometry()
        measured = np.zeros(14)
        measured[0] = 0.0001
        pose = solver.poses(measured)
        state = np.r_[pose[0], 0.5, np.zeros(6), pose[1], 0.5, np.zeros(6)]
        obs = Observation(
            "lagged",
            time.monotonic(),
            {},
            state,
            measured,
            np.zeros(14),
            np.array([0.5, 0.5]),
            {},
            {"execution_start_target": [0.0] * 14},
        )
        identity = CycleIdentity("run", 1, 0, "lagged", "lagged-request")
        prediction = SyntheticPolicy(delay=0).predict(
            obs.request("task", 0), identity.request_id
        )
        with tempfile.TemporaryDirectory() as folder:
            plan = PrefixPreparer(solver, LoopConfig()).prepare(
                obs, prediction, identity, Path(folder) / "plan"
            )
        np.testing.assert_array_equal(plan.seed, np.zeros(14))
        self.assertEqual(plan.checks["measured_seed"][0], 0.0001)
        failures, _, _ = kinematic_checks(
            plan.joints, plan.seed, plan.seed_velocity, LoopConfig(), solver.bounds
        )
        self.assertEqual(failures, [])
        self.assertLess(abs(plan.joints[0, 0]), 0.0000041)

    def test_native_prefix_is_50_ticks_not_834(self):
        limits = json.loads(Path("config/planning.json").read_text())
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
        pose = solver.poses(q)
        state = np.r_[pose[0], 0.5, np.zeros(6), pose[1], 0.5, np.zeros(6)]
        actions = np.tile(np.r_[pose[0], 0.5, pose[1], 0.5], (1, 50, 1))
        actions[0, :, 0] += 0.0001
        obs = Observation(
            "sample",
            time.monotonic(),
            {},
            state,
            q,
            np.zeros(14),
            np.array([0.5, 0.5]),
            {},
            {},
        )
        identity = CycleIdentity("run", 1, 0, "sample", "request")
        # Integration includes native startup and independent geometry checking.
        c = LoopConfig(prepare_timeout_s=2.0, max_prediction_age_s=4.0)
        with tempfile.TemporaryDirectory() as folder:
            segment = PrefixPreparer(solver, c).prepare(
                obs,
                dict(actions=actions, request_id="request"),
                identity,
                Path(folder) / "prefix",
            )
            checks = json.loads(
                (Path(folder) / "prefix/reference_qp_checks.json").read_text()
            )
            self.assertEqual(checks["simulated_ticks"], 50)
            self.assertEqual(len(segment.joints), 50)
            self.assertGreater(np.max(np.abs(segment.joints - q)), 0)
            self.assertEqual(segment.checks["time_policy"], "original_30Hz_no_retiming")
            self.assertEqual(len(segment.brake_paths), 50)


if __name__ == "__main__":
    unittest.main()
