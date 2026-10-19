import json
from pathlib import Path
import sys
import unittest
import numpy as np

from robot_control.common import ROOT, load_config
from robot_control.command_bridge import CommandBridge, Prediction, ControlFault, current_action
from robot_control.hardware_backend import (
    SimulatedBackend,
    HardwareBackend,
    encode_targets,
    readiness_audit,
)
from robot_control.run_control import simulation


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.limits = json.loads((ROOT / "config/control_simulation.json").read_text())
        self.initial = np.array(
            [
                0.5,
                0.1,
                0.7,
                0,
                0,
                0,
                0.5,
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
                0.5,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
            float,
        )
        self.backend = SimulatedBackend(self.initial)
        self.bridge = CommandBridge(self.backend, self.limits)
        self.bridge.connect(10.0)

    def submit(self, actions=None, checks=True, now=10.0):
        actions = (
            np.tile(current_action(self.initial), (50, 1))
            if actions is None
            else actions
        )
        self.bridge.request("r", 10.0, 10.0)
        self.bridge.submit(Prediction("r", 10.0, actions, checks), now)

    def test_success_finite_prefix_holds_and_stops(self):
        report = simulation(self.limits)
        self.assertEqual(report["state"], "STOPPED")
        self.assertTrue(all(report["stop_receipts"].values()))
        events = report["events"]
        self.assertTrue(any(e["event"] == "chunk_complete" for e in events))
        writes = [e for e in events if e["event"] == "submitted"]
        self.assertTrue(any(e["mode"] == "WAITING" for e in writes))
        self.assertTrue(any(e["mode"] == "EXECUTING" for e in writes))
        self.assertLess(max(e["target"][0] for e in writes), 0.501)
        self.assertEqual(report["hardware_commands_sent"], 0)

    def test_simulation_fault_scenarios_record_terminal_state(self):
        for fault in [
            "owner_lost",
            "gripper_owner_lost",
            "disconnect",
            "stale_feedback",
            "tracking_error",
            "ik_failure",
            "right_write_failure",
            "gripper_reject",
            "stop_unconfirmed",
        ]:
            with self.subTest(fault=fault):
                result = simulation(self.limits, fault)
                self.assertIn(result["state"], ["STOPPED", "FAULT_UNCONFIRMED"])
                self.assertEqual(result["hardware_commands_sent"], 0)
                self.assertTrue(result["fault_reason"])

    def test_invalid_tail_rejected_before_first_send(self):
        a = np.tile(current_action(self.initial), (50, 1))
        a[-1, 13] = -0.2
        with self.assertRaisesRegex(ControlFault, "gripper"):
            self.submit(a)
        self.assertEqual(self.backend.write_count, 0)
        self.assertEqual(self.bridge.state, "STOPPED")

    def test_constraints(self):
        for name, mutate in [
            ("workspace", lambda a: a.__setitem__((slice(None), 0), 2)),
            ("position_step", lambda a: a.__setitem__((0, 0), 0.52)),
            ("linear_speed", lambda a: a.__setitem__((0, 0), 0.505)),
            ("linear_acceleration", lambda a: a.__setitem__((0, 0), 0.502)),
            ("gripper_speed", lambda a: a.__setitem__((0, 6), 0.6)),
        ]:
            with self.subTest(name=name):
                a = np.tile(current_action(self.initial), (50, 1))
                mutate(a)
                with self.assertRaisesRegex(ControlFault, name):
                    self.bridge.validate_sequence(a, current_action(self.initial))

    def test_nonfinite_and_shape_rejected(self):
        a = np.tile(current_action(self.initial), (50, 1))
        a[1, 0] = np.nan
        with self.assertRaises(ControlFault):
            self.submit(a)
        self.assertEqual(self.backend.write_count, 0)

    def test_prediction_mismatch_stops(self):
        self.bridge.request("r", 10, 10)
        with self.assertRaisesRegex(ControlFault, "identity"):
            self.bridge.submit(
                Prediction(
                    "wrong", 10, np.tile(current_action(self.initial), (50, 1)), True
                ),
                10,
            )
        self.assertEqual(self.bridge.state, "STOPPED")

    def test_checks_failed_cannot_be_clipped_or_overridden(self):
        with self.assertRaisesRegex(ControlFault, "checks_failed"):
            self.submit(checks=False)
        self.assertEqual(self.backend.write_count, 0)

    def test_stale_prediction_during_execution(self):
        self.limits["prediction_max_age_s"] = 0.01
        self.submit()
        for now in [10.0, 10.005, 10.010, 10.015]:
            self.bridge.tick(now)
        self.assertEqual(self.bridge.fault_reason, "prediction_expired")
        count = self.backend.write_count
        self.bridge.tick(11.0)
        self.assertEqual(count, self.backend.write_count)
        self.assertIsNone(self.bridge.actions)

    def test_no_resume_or_old_prediction_after_stop(self):
        self.submit()
        self.bridge.fail(10.0, "test")
        with self.assertRaises(ControlFault):
            self.bridge.connect(11.0)
        with self.assertRaises(ControlFault):
            self.bridge.submit(Prediction("r", 10, np.zeros((50, 14)), True), 11)
        self.assertIsNone(self.bridge.pending)

    def test_owner_and_gripper_ownership_lost(self):
        for fault in ["owner_lost", "gripper_owner_lost"]:
            with self.subTest(fault=fault):
                backend = SimulatedBackend(self.initial)
                b = CommandBridge(backend, self.limits)
                b.connect(10.0)
                backend.fault = fault
                b.tick(10.0)
                self.assertEqual(b.fault_reason, "ownership_lost")
                self.assertEqual(backend.write_count, 0)

    def test_read_and_ik_faults_stop_before_write(self):
        for fault in [
            "disconnect",
            "stale_feedback",
            "tracking_error",
            "ik_failure",
            "nan_feedback",
        ]:
            with self.subTest(fault=fault):
                backend = SimulatedBackend(self.initial)
                b = CommandBridge(backend, self.limits)
                b.connect(10.0)
                backend.fault = fault
                b.tick(10.0)
                self.assertEqual(b.state, "STOPPED")
                self.assertEqual(backend.write_count, 0)

    def test_partial_left_sent_right_failure_stops_all(self):
        self.submit()
        self.backend.fault = "right_write_failure"
        self.bridge.tick(10.0)
        self.assertEqual(self.backend.write_count, 1)
        self.assertEqual(self.backend.stop_calls, ["left", "right", "grippers"])
        self.bridge.tick(10.005)
        self.assertEqual(self.backend.write_count, 1)

    def test_gripper_rejected_stops_both_arms(self):
        self.submit()
        self.backend.fault = "gripper_reject"
        self.bridge.tick(10.0)
        self.assertEqual(self.bridge.fault_reason, "partial_or_rejected_submission")
        self.assertTrue(all(self.bridge.stop_receipts.values()))

    def test_stop_not_acknowledged_never_claims_stopped(self):
        self.backend.fault = "stop_unconfirmed"
        self.bridge.fail(10.0, "test")
        self.assertEqual(self.bridge.state, "FAULT_UNCONFIRMED")
        self.assertIsNotNone(self.backend.token)
        self.assertFalse(any(self.bridge.stop_receipts.values()))

    def test_release_failure_is_unconfirmed(self):
        self.backend.fault = "release_failure"
        self.bridge.fail(10.0, "test")
        self.assertEqual(self.bridge.state, "FAULT_UNCONFIRMED")

    def test_deadline_no_catchup_burst(self):
        self.submit()
        self.bridge.tick(10.1)
        self.assertEqual(self.bridge.fault_reason, "control_deadline_missed")
        self.assertEqual(self.backend.write_count, 0)

    def test_independent_mock_watchdog_without_bridge_ticks(self):
        self.backend.watchdog(10.2)
        self.assertIsNone(self.backend.token)
        self.assertEqual(self.backend.events[-1]["event"], "independent_watchdog")

    def test_rpc_timeout_while_hold_continues(self):
        report = simulation(self.limits, "rpc_timeout")
        self.assertEqual(report["fault_reason"], "rpc_timeout")
        self.assertGreater(report["submitted_ticks"], 300)
        self.assertFalse(any(e.get("mode") == "EXECUTING" for e in report["events"]))

    def test_acquisition_failure_does_not_stop_someone_else(self):
        backend = SimulatedBackend(self.initial, fault="ownership_busy")
        bridge = CommandBridge(backend, self.limits)
        with self.assertRaises(ControlFault):
            bridge.connect(10.0)
        self.assertEqual(backend.stop_calls, [])
        self.assertEqual(backend.write_count, 0)

    def test_ros_message_encoding_no_publishers(self):
        a = current_action(self.initial)
        a[3:6] = [0.2, -0.1, 0.5]
        message, l, r = encode_targets(a, np.arange(14) * 0.01, 1234567890)
        self.assertEqual(message.header.frame_id, "base_link")
        self.assertEqual(list(message.available), [True, True] + [False] * 9)
        self.assertAlmostEqual(message.left_hand_pose.position.x, 0.5)
        self.assertAlmostEqual(l.data, 0.5)
        self.assertAlmostEqual(r.data, 0.5)
        self.assertEqual(len(message.joint_state.position), 14)
        with self.assertRaises(ValueError):
            encode_targets(np.full(14, np.nan), np.zeros(14), 123)

    def test_negative_actual_uses_existing_target_for_claim(self):
        from robot_control.hardware_backend import gripper_claim_readiness

        reply = {
            "ok": True,
            "data": {
                "reserved": False,
                "healthy": True,
                "actual": [-0.02, -0.021],
                "applied_target": [0.0, 0.0],
                "calibration": [[0.0, 1.6], [0.0, 1.6]],
                "motor_feedback_age": [0.001, 0.001],
                "status_age_sec": 0.01,
            },
        }
        original = reply["data"]["actual"].copy()
        result = gripper_claim_readiness(reply, 0.2)
        self.assertTrue(result["driver_claim_preconditions_passed"])
        self.assertEqual(result["reasons"], [])
        result = gripper_claim_readiness(reply, 0.2, [0.0, 0.0])
        self.assertTrue(result["driver_claim_preconditions_passed"])
        self.assertEqual(result["claim_seed_existing_target"], [0.0, 0.0])
        self.assertEqual(reply["data"]["actual"], original)
        reply["data"]["actual"][1] = 0.3
        result = gripper_claim_readiness(reply, 0.2, [0.1, -0.1])
        self.assertTrue(result["driver_claim_preconditions_passed"])
        self.assertEqual(result["reasons"], [])

    def test_claim_health_does_not_hide_old_motor_feedback(self):
        from robot_control.hardware_backend import gripper_claim_readiness

        reply = {
            "ok": True,
            "data": {
                "reserved": False,
                "healthy": True,
                "actual": [0.5, 0.5],
                "applied_target": [0.5, 0.5],
                "calibration": [[0.0, 1.6], [0.0, 1.6]],
                "motor_feedback_age": [1.0, 0.001],
                "status_age_sec": 0.01,
            },
        }
        result = gripper_claim_readiness(reply, 0.2, [0.0, 0.0])
        self.assertTrue(result["driver_claim_preconditions_passed"])
        self.assertIn("gripper_feedback_stale_or_unknown", result["reasons"])
        self.assertFalse(result["claim_request_sent"])

    def test_installed_wrapper_claim_uses_command_not_actual(self):
        import importlib.util, tempfile
        from types import SimpleNamespace

        import os
        path = os.environ.get("FORCEVLA_GRIPPER_RULES_PATH", "/opt/kernelmind/apex_tool/install/lib/python3.12/site-packages/dm_gripper_py/web_gripper.py")
        spec = importlib.util.spec_from_file_location("isolated_gripper_rules", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # Only pure Gate and an unbound method with dummy motors are used.
        # No WebGripper constructor, ROS initialization, CAN or motor command.
        for velocity, position, expected in [
            (0.0, -0.034, True),
            (0.1, -0.034, True),
            (0.0, 0.49, True),
            (0.1, 0.49, True),
        ]:
            with (
                self.subTest(velocity=velocity, position=position),
                tempfile.TemporaryDirectory() as directory,
            ):
                gate = module.Gate(Path(directory) / "lease.json", clock=lambda: 10.0)
                motor = SimpleNamespace(
                    getVelocity=lambda: velocity, getPosition=lambda: position
                )
                fake = SimpleNamespace(
                    gate=gate,
                    node=SimpleNamespace(q1=0.0, q2=0.0),
                    feedback=lambda: ([position / 1.6] * 2, [0.001] * 2, [1, 1], True),
                    read_motors=lambda: [(motor, 0.0, 1.6), (motor, 0.0, 1.6)],
                    fault_reason=lambda *args: "",
                    apply=lambda: None,
                    publish_status=lambda: None,
                )
                req = SimpleNamespace(
                    data=json.dumps(
                        {"action": "claim", "boot_id": gate.boot, "request_id": "unit"}
                    )
                )
                module.WebGripper.command(fake, req)
                self.assertEqual(gate.reserved, expected)
                if expected:
                    self.assertEqual(gate.issued, [0.0, 0.0])
                    self.assertEqual(gate.target, [0.0, 0.0])

    def test_live_backend_cannot_be_armed(self):
        backend = HardwareBackend()
        with self.assertRaisesRegex(ControlFault, "unavailable"):
            backend.acquire(0)
        with self.assertRaises(ControlFault):
            backend.write(np.zeros(14), "token", 0)
        config = load_config()
        for k in list(config):
            if k.endswith("_verified"):
                config[k] = True
        report = readiness_audit(config, live=False)
        self.assertEqual(report["status"], "blocked_hardware_execution")
        self.assertIn("hardware_transport_unavailable", report["blocking_reasons"])


if __name__ == "__main__":
    unittest.main()
