import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from robot_control.closed_loop import ComputeLane
from robot_control.closed_loop_contracts import Cancellation, CycleIdentity, LoopConfig
from robot_control.closed_loop_native import NativeSimulatedIO, create_session
from robot_control.closed_loop_observation import ObservationPending
from robot_control.closed_loop_planning import PrefixPreparer
from robot_control.closed_loop_simulation import (
    SimulatedObservationSource,
    SyntheticGeometry,
    SyntheticPolicy,
)
from robot_control.command_bridge import ControlFault


class SessionTests(unittest.TestCase):
    def setUp(self):
        original_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.0002)
        self.addCleanup(sys.setswitchinterval, original_interval)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = LoopConfig()
        self.io = NativeSimulatedIO(
            self.config, Path(self.tmp.name) / "native", SyntheticGeometry().bounds
        )
        self.session = create_session(
            self.io,
            SyntheticGeometry().bounds,
            self.config,
            Path(self.tmp.name) / "execution",
        )
        self.addCleanup(self.session.close)
        self.supervision = threading.Event()

        def supervise():
            while not self.supervision.wait(0.01):
                self.session.supervise()

        self.supervisor = threading.Thread(target=supervise, daemon=True)
        self.supervisor.start()
        self.addCleanup(self.supervision.set)
        self.source = SimulatedObservationSource(self.io, self.config)
        self.preparer = PrefixPreparer(SyntheticGeometry(), self.config)
        self.policy = SyntheticPolicy()
        self.compute = ComputeLane("simulate", self.config, {})
        self.compute.ready(self.session.supervise)
        self.addCleanup(self.compute.close)
        self.cancel = Cancellation()

    def plan(self, cycle):
        end = time.monotonic() + 0.5
        while True:
            try:
                obs = self.source.read_after(self.session.barrier, None)
                break
            except ObservationPending:
                if time.monotonic() > end:
                    raise
                time.sleep(0.005)
        ident = CycleIdentity("run", cycle, 0, obs.id, "request" + str(cycle))
        result = self.compute.call(
            "infer",
            (obs.request("task", 0), ident.request_id),
            2.0,
            self.session.supervise,
            self.cancel,
            0,
        )
        return self.compute.call(
            "prepare",
            (obs, result, ident, Path(self.tmp.name) / f"cycle-{cycle}"),
            1.0,
            self.session.supervise,
            self.cancel,
            0,
        )

    def test_same_session_three_cycles_with_changed_feedback_and_gripper(self):
        self.session.acquire("test")
        positions = []
        for cycle in range(1, 4):
            plan = self.plan(cycle)
            result = self.session.execute(plan, self.cancel)
            self.assertEqual(result["executed_model_steps"], 3)
            self.assertEqual(self.session.phase, "HOLDING")
            positions.append(self.io.q[0])
        self.assertTrue(positions[0] < positions[1] < positions[2], positions)
        self.assertGreater(self.io.grip[0], 0.5)
        self.assertEqual(self.io.claims, 1)
        report = self.session.stop("normal_completion")
        self.assertTrue(report["confirmed"], report)
        self.session.release()
        self.assertEqual(self.io.releases, 1)
        self.assertEqual(self.io.input, 0)

    def test_cancel_latches_and_prevents_replay(self):
        self.session.acquire("test")
        plan = self.plan(1)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(self.session.execute, plan, self.cancel)
            deadline = time.monotonic() + 1
            while self.session.index < 3 and time.monotonic() < deadline:
                time.sleep(0.001)
            self.cancel.cancel()
            with self.assertRaises(ControlFault):
                future.result(timeout=2)
        report = self.session.stop("user_cancelled")
        self.assertTrue(report["confirmed"], report)
        with self.assertRaises(ControlFault):
            self.session.execute(plan, Cancellation())
        self.session.release()

    def test_stop_during_claim_cannot_resume_normal_hold(self):
        self.io.gripper_delay = 0.12
        with ThreadPoolExecutor(1) as pool:
            acquiring = pool.submit(self.session.acquire, "test")
            deadline = time.monotonic() + 1
            while (
                self.session.backend.state != "ACQUIRING"
                and time.monotonic() < deadline
            ):
                time.sleep(0.001)
            report = self.session.stop("user_cancelled")
            with self.assertRaises(ControlFault):
                acquiring.result(timeout=2)
        self.assertTrue(report["confirmed"], report)
        self.assertIsNone(self.session.barrier)
        self.assertEqual(self.session.phase, "STOP_CONFIRMED")
        self.session.release()

    def test_log_failure_does_not_resume_or_claim_stop_evidence(self):
        self.session.acquire("test")
        self.session.audit.error = "injected_disk_error"
        deadline = time.monotonic() + 2
        while (
            self.session.phase not in ("STOP_CONFIRMED", "STOP_UNCONFIRMED")
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertEqual(self.session.phase, "STOP_UNCONFIRMED")
        self.assertEqual(self.io.grip_state, "held")
        # The fixture clears only the injected journal error for cleanup; the
        # session remains latched and still cannot return to normal execution.
        self.session.audit.error = None

    def test_cancel_during_settling_does_not_start_another_segment(self):
        self.session.acquire("test")
        plan = self.plan(1)
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(self.session.execute, plan, self.cancel)
            deadline = time.monotonic() + 2
            while self.session.phase != "SETTLING" and time.monotonic() < deadline:
                time.sleep(0.002)
            self.assertEqual(self.session.phase, "SETTLING")
            self.cancel.cancel()
            with self.assertRaises(ControlFault):
                running.result(timeout=2)
        report = self.session.stop("user_cancelled")
        self.assertTrue(report["confirmed"], report)
        self.assertEqual(self.session.index, len(plan.joints))
        self.session.release()

    def test_late_gripper_receipt_after_cancel_cannot_resume(self):
        self.session.acquire("test")
        plan = self.plan(1)
        self.io.gripper_delay = 0.08
        with ThreadPoolExecutor(1) as pool:
            running = pool.submit(self.session.execute, plan, self.cancel)
            deadline = time.monotonic() + 1
            while not self.session.gripper_pending and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(self.session.gripper_pending)
            self.cancel.cancel()
            with self.assertRaises(ControlFault):
                running.result(timeout=2)
        report = self.session.stop("user_cancelled")
        self.assertTrue(report["confirmed"], report)
        self.assertEqual(self.session.phase, "STOP_CONFIRMED")
        self.assertEqual(self.io.grip_state, "held")
        self.session.release()

    def test_supervisor_loss_stops_during_idle_hold(self):
        self.session.acquire("test")
        self.supervision.set()
        deadline = time.monotonic() + 2
        while (
            self.session.phase not in ("STOP_CONFIRMED", "STOP_UNCONFIRMED")
            and time.monotonic() < deadline
        ):
            time.sleep(0.01)
        self.assertIn(
            self.session.stop_reason,
            ("supervisor_expired", "native_supervisor_expired"),
        )
        self.assertEqual(self.session.phase, "STOP_CONFIRMED", self.session.status())
        self.session.release()

    def test_failed_arm_never_reports_completed_segment(self):
        self.session.acquire("test")
        plan = self.plan(1)
        self.io.fail_side = "right"
        with self.assertRaises(ControlFault):
            self.session.execute(plan, self.cancel)
        result = self.session.stop("arm_failed")
        self.assertFalse(result["confirmed"])
        self.assertLess(self.session.executed_steps, 3)


if __name__ == "__main__":
    unittest.main()
