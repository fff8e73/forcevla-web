import copy
import importlib.util
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import ExecutionFeedback, JointExecutionBackend
from robot_control.joint_source_guard import JointEnvelope


class FakeIO:
    """Real installed pure gripper Gate, ideal joint plant; no ROS or motors."""

    def __init__(self, folder, clock):
        self.clock = clock
        spec = importlib.util.spec_from_file_location(
            "isolated_gate",
            "/opt/kernelmind/apex_tool/install/lib/python3.12/site-packages/dm_gripper_py/web_gripper.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.gate = module.Gate(Path(folder) / "gripper.json", clock=clock)
        self.q = np.zeros(14)
        self.v = np.zeros(14)
        self.target = self.q.copy()
        self.input = 0
        self.mode = 1
        self.fail_side = None
        self.grip_fail = None
        self.stale = False
        self.unexpected = False
        self.old_sample = None
        self.events = []
        self.production = False
        self.claim_timeout = False
        self.select_timeout = False

    def gripper_status(self):
        g = self.gate.status()
        issued = g["issued"] if g["reserved"] else [0.5, 0.5]
        return dict(
            g,
            actual=list(issued),
            applied_target=list(issued),
            healthy=True,
            motor_feedback_age=[0.0, 0.0],
            motor_velocities=[0.0, 0.0],
            calibration=[[0.0, 1.6], [0.0, 1.6]],
        )

    def feedback(self):
        t = self.old_sample if self.old_sample is not None else self.clock()
        return ExecutionFeedback(
            t - 1 if self.stale else t,
            1.0 if self.stale else 0.0,
            self.q.copy(),
            self.v.copy(),
            self.target.copy(),
            t,
            self.input,
            True,
            self.mode,
            copy.deepcopy(self.gripper_status()),
            t,
            self.unexpected,
        )

    def bind_stationary_target(self, q, grip):
        self.seed = q.copy()
        self.grip_seed = grip.copy()

    def validate_targets(self, q, grip):
        if self.production and (
            not np.array_equal(q, self.seed) or not np.array_equal(grip, self.grip_seed)
        ):
            raise ControlFault("production_stage_B_stationary_only")

    def select_input(self, value):
        self.events.append(("input", value))
        self.input = value
        if self.select_timeout:
            raise TimeoutError("select reply lost")

    def gripper_command(self, request):
        action = request["action"]
        self.events.append(("gripper", action))
        if self.grip_fail == action:
            raise TimeoutError("injected gripper failure")
        self.gate.command(request, [0.5, 0.5], True)
        if self.claim_timeout and action == "claim":
            raise TimeoutError("claim reply lost after accepted")
        return self.gripper_status()

    def publish_arm(self, side, q):
        self.events.append(("arm", side, list(q)))
        if self.fail_side == side:
            raise ConnectionError(side + " send failed")
        i = 0 if side == "left" else 7
        self.q[i : i + 7] = q
        self.target[i : i + 7] = q

    def heartbeat(self, request):
        self.gate.heartbeat(request, True)

    def close(self):
        pass


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.now = 10.0
        self.io = FakeIO(self.tmp.name, lambda: self.now)
        self.b = JointExecutionBackend(
            self.io,
            np.tile([-2, 2], (14, 1)),
            np.ones(14),
            Path(self.tmp.name) / "execution.json",
            clock=lambda: self.now,
        )

    def acquire(self):
        self.token = self.b.acquire("test")
        return self.token

    def command(self, q=None, observed=10.0, seq=0):
        self.now += 0.02
        return JointEnvelope(
            self.token, seq, self.now, observed, np.zeros(14) if q is None else q
        )

    def settle(self):
        for _ in range(15):
            self.now += 0.02
            self.b.poll()

    def test_success_preserves_target_and_stops_with_new_feedback(self):
        self.io.q[:] = 0.3
        self.io.target[:] = 0.3
        self.acquire()
        self.b.write(self.command(np.full(14, 0.3)))
        self.b.trip("manual")
        self.assertEqual(self.b.state, "STOPPING")
        self.settle()
        self.assertEqual(self.b.state, "HOLD_CONFIRMED")
        self.assertFalse(self.b.stop_report["hardware_stop_verified"])
        self.b.release(self.token)
        self.assertEqual(self.io.input, 0)
        self.assertEqual(self.b.state, "RELEASED")
        self.assertFalse(self.io.gate.reserved)

    def test_partial_right_failure_attempts_gripper_hold_and_latches(self):
        self.acquire()
        self.io.fail_side = "right"
        q = np.full(14, 0.001)
        with self.assertRaises(ConnectionError):
            self.b.write(self.command(q))
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")
        self.assertIn(("gripper", "hold"), self.io.events)
        self.assertTrue(self.io.gate.reserved)
        # Do not roll the already-submitted left arm back to its previous goal.
        np.testing.assert_array_equal(self.b.last_q[:7], q[:7])
        before = len(self.io.events)
        with self.assertRaises(ControlFault):
            self.b.write(self.command(q, seq=1))
        self.assertEqual(len(self.io.events), before)

    def test_claim_reply_loss_reconciles_own_gripper_and_keeps_reservation(self):
        self.io.claim_timeout = True
        with self.assertRaises(TimeoutError):
            self.acquire()
        self.assertEqual(self.io.gate.state, "held")
        self.assertTrue(self.io.gate.reserved)
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")
        self.assertFalse(any(x[0] == "arm" for x in self.io.events))

    def test_selection_reply_loss_reconciles_without_retry(self):
        self.io.select_timeout = True
        with self.assertRaises(TimeoutError):
            self.acquire()
        self.assertEqual(sum(x == ("input", 3) for x in self.io.events), 1)
        self.assertTrue(self.b.stop_report["arms_hold_sent"])
        self.assertEqual(self.io.gate.state, "held")

    def test_busy_owner_no_commands(self):
        self.io.input = 1
        with self.assertRaises(ControlFault):
            self.acquire()
        self.assertEqual(self.io.events, [])

    def test_unknown_gripper_owner_never_held_or_released(self):
        self.acquire()
        self.io.gate.session = "someone-else"
        self.io.events.clear()
        with self.assertRaises(ControlFault):
            self.b.write(self.command())
        self.assertFalse(any(e[0] == "gripper" for e in self.io.events))
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_source_changed_no_arm_hold_into_other_source(self):
        self.acquire()
        self.io.input = 1
        self.io.events.clear()
        self.b.poll()
        self.assertFalse(any(e[0] == "arm" for e in self.io.events))
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_stale_feedback_still_attempts_independent_gripper_hold(self):
        self.acquire()
        self.io.stale = True
        self.b.poll()
        self.assertEqual(self.io.gate.state, "held")
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_duplicate_and_stale_commands_latch(self):
        self.acquire()
        envelope = self.command()
        self.b.write(envelope)
        with self.assertRaisesRegex(Exception, "duplicate"):
            self.b.write(envelope)
        self.assertNotEqual(self.b.state, "ACTIVE")

    def test_old_observation_not_refreshed_by_heartbeat(self):
        self.acquire()
        self.b.heartbeat(self.token)
        with self.assertRaisesRegex(Exception, "prediction_stale"):
            self.b.write(self.command(observed=1.0))

    def test_wrong_token_cannot_trip_owner(self):
        self.acquire()
        e = self.command()
        with self.assertRaises(ControlFault):
            self.b.write(
                JointEnvelope(
                    "wrong",
                    e.sequence,
                    e.command_time,
                    e.observation_time,
                    e.joint_positions,
                )
            )
        self.assertEqual(self.b.state, "ACTIVE")

    def test_gripper_failure_after_arms_trips_all(self):
        self.acquire()
        self.io.grip_fail = "target"
        with self.assertRaises(TimeoutError):
            self.b.write(self.command(), [0.51, 0.5])
        self.assertEqual(self.io.gate.state, "held")
        self.assertNotEqual(self.b.state, "ACTIVE")

    def test_no_stopped_claim_from_repeated_old_sample(self):
        self.acquire()
        self.b.trip("test")
        self.io.old_sample = self.now
        self.settle()
        self.assertNotEqual(self.b.state, "HOLD_CONFIRMED")
        self.now += 2.0
        self.b.poll()
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_nonzero_velocity_never_confirms_hold(self):
        self.acquire()
        self.b.trip("test")
        self.io.v[:] = 0.1
        self.settle()
        self.assertEqual(self.b.state, "STOPPING")
        self.now += 2
        self.b.poll()
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_failed_release_retains_fault_and_none_input(self):
        self.acquire()
        self.b.trip("test")
        self.settle()
        self.io.grip_fail = "release"
        with self.assertRaises(TimeoutError):
            self.b.release(self.token)
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")
        self.assertTrue(self.io.gate.reserved)
        self.assertEqual(self.io.input, 0)

    def test_process_restart_requires_review(self):
        self.acquire()
        b = JointExecutionBackend(
            self.io, np.tile([-2, 2], (14, 1)), np.ones(14), self.b.journal
        )
        with self.assertRaises(ControlFault):
            b.acquire("new")
        self.assertEqual(b.state, "REVIEW_REQUIRED")

    def test_production_motion_rejected_before_any_target_send(self):
        self.io.production = True
        self.b.start_watchdog()
        self.addCleanup(self.b.close)
        self.acquire()
        self.io.events.clear()
        with self.assertRaisesRegex(ControlFault, "stationary_only"):
            self.b.write(self.command(np.full(14, 0.001)))
        arms = [e for e in self.io.events if e[0] == "arm"]
        self.assertTrue(all(np.array_equal(e[2], np.zeros(7)) for e in arms))

    def test_continuous_commands_do_not_renew_lease(self):
        self.acquire()
        for i in range(12):
            self.b.write(self.command(seq=i))
        self.now += 0.02
        self.b.poll()
        self.assertNotEqual(self.b.state, "ACTIVE")

    def test_background_watchdog_without_caller_ticks(self):
        io = FakeIO(self.tmp.name, time.monotonic)
        b = JointExecutionBackend(
            io,
            np.tile([-2, 2], (14, 1)),
            np.ones(14),
            Path(self.tmp.name) / "threaded.json",
            lease_s=0.08,
        )
        b.acquire("thread-test")
        b.start_watchdog(0.01)
        try:
            deadline = time.monotonic() + 0.8
            while time.monotonic() < deadline:
                with b.lock:
                    if b.state != "ACTIVE":
                        break
                time.sleep(0.01)
            self.assertNotEqual(b.state, "ACTIVE")
            self.assertEqual(io.gate.state, "held")
        finally:
            b.close()

    def test_log_write_failure_still_requests_hold(self):
        self.acquire()

        def failed_record(*args, **kwargs):
            raise OSError("disk full")

        self.b.record = failed_record
        self.b.trip("test_disk_full")
        self.assertEqual(self.io.gate.state, "held")
        self.assertTrue(hasattr(self.b, "journal_error"))
        self.settle()
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_left_failure_does_not_skip_right_hold(self):
        self.acquire()
        self.io.fail_side = "left"
        self.b.trip("test_left_failure")
        self.assertTrue(any(e[:2] == ("arm", "right") for e in self.io.events))
        self.assertEqual(
            self.b.stop_report["arm_hold_submissions"], {"left": False, "right": True}
        )
        self.assertEqual(self.io.gate.state, "held")

    def test_gripper_hold_timeout_keeps_ownership(self):
        self.acquire()
        self.io.grip_fail = "hold"
        self.b.trip("stop")
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")
        self.assertTrue(self.io.gate.reserved)
        with self.assertRaises(ControlFault):
            self.b.release(self.token)

    def test_driver_restart_does_not_touch_new_gripper_owner(self):
        self.acquire()
        self.io.gate.boot = "new-boot"
        self.io.events.clear()
        self.b.poll()
        self.assertFalse(any(e[0] == "gripper" for e in self.io.events))
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_mode_change_latches(self):
        self.acquire()
        self.io.mode = 3
        self.b.poll()
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")

    def test_external_publisher_blocks_all_arm_sends(self):
        self.acquire()
        self.io.events.clear()
        self.io.unexpected = True
        self.b.poll()
        self.assertFalse(any(e[0] == "arm" for e in self.io.events))
        self.assertEqual(self.b.state, "FAULT_UNCONFIRMED")


if __name__ == "__main__":
    unittest.main()
