import tempfile
import threading
import time
import unittest
import uuid

import numpy as np

from robot_control.closed_loop_contracts import LoopConfig
from robot_control.closed_loop_native import NativeClient
from robot_control.closed_loop_planning import brake_path
from robot_control.command_bridge import ControlFault


class NativeExecutorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.config = LoopConfig()
        self.client = NativeClient(self.config, self.tmp.name, simulated=True)
        self.addCleanup(self.client.close)
        self.supervisor_stop = threading.Event()

        def supervise():
            while not self.supervisor_stop.wait(0.01):
                self.client.supervise(time.monotonic())

        threading.Thread(target=supervise, daemon=True).start()
        self.addCleanup(self.supervisor_stop.set)
        self.client.command(
            "bind",
            q=[0.0] * 14,
            bounds=[[-2.0, 2.0]] * 14,
            boot="synthetic",
            session=str(uuid.uuid4()),
        )
        self.client.command("activate")

    def plan(self, ident="test-plan", seed=None):
        seed = np.zeros(14) if seed is None else np.asarray(seed)
        t = np.arange(1, 51) / 50
        q = np.tile(seed, (50, 1))
        q[:, 0] += 0.00008 * (10 * t**3 - 15 * t**4 + 6 * t**5)
        v = np.diff(np.vstack((seed, q)), axis=0) / 0.002
        brakes = [brake_path(row, vel, 0.002, 1.0).tolist() for row, vel in zip(q, v)]
        return dict(
            id=ident,
            seed=seed.tolist(),
            seed_velocity=[0.0] * 14,
            observed_at=time.monotonic(),
            q=q.tolist(),
            brakes=brakes,
        )

    def wait(self, state):
        end = time.monotonic() + 2
        while time.monotonic() < end:
            status = self.client.status()
            if status["state"] == state:
                return status
            if status["state"] == "ABORTED" and state != "ABORTED":
                self.fail(status)
            time.sleep(0.003)
        self.fail(self.client.status())

    def test_two_finite_segments_with_original_500hz_deadlines(self):
        for i in range(2):
            seed = self.client.status()["last_q"]
            self.client.command("arm", **self.plan("plan-" + str(i), seed))
            result = self.wait("HOLDING")
            self.assertEqual(result["sent_ticks"], 50)
            self.assertLessEqual(result["max_submission_lateness_s"], 0.0015)
        self.assertAlmostEqual(result["last_q"][0], 0.00016)
        with self.assertRaisesRegex(ControlFault, "native_replay"):
            self.client.command("arm", **self.plan("plan-1", result["last_q"]))
        self.client.command("deactivate")

    def test_invalid_velocity_and_old_observation_rejected(self):
        plan = self.plan()
        plan["q"][0][0] = 0.009
        with self.assertRaisesRegex(ControlFault, "native_derivative_bound"):
            self.client.command("arm", **plan)
        plan = self.plan("old")
        plan["observed_at"] -= 3
        with self.assertRaisesRegex(ControlFault, "native_age_budget"):
            self.client.command("arm", **plan)
        self.assertEqual(self.client.status()["sent_ticks"], 0)
        self.client.command("deactivate")

    def test_cancellation_latches_and_preserves_checked_braking(self):
        self.client.command("arm", **self.plan())
        deadline = time.monotonic() + 1
        while self.client.status()["sent_ticks"] < 5 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.client.command("abort", reason="test_cancel")
        result = self.wait("ABORTED")
        self.assertLess(result["sent_ticks"], 50)
        self.assertTrue(result["latched"])
        with self.assertRaisesRegex(ControlFault, "native_requires_hold"):
            self.client.command("arm", **self.plan("after-cancel", result["last_q"]))

    def test_parent_socket_loss_stops_without_python_controller(self):
        self.client.command("arm", **self.plan())
        deadline = time.monotonic() + 1
        while self.client.status()["sent_ticks"] < 5 and time.monotonic() < deadline:
            time.sleep(0.001)
        self.client.finished.set()
        self.client.thread.join(timeout=0.5)
        self.client.socket.close()
        self.client.process.wait(timeout=3)
        report = __import__("json").loads(
            (self.client.folder / "native-result.json").read_text()
        )
        self.assertEqual(report["reason"], "native_parent_disconnected")
        self.assertTrue(report["latched"])
        self.assertLess(report["sent_ticks"], 50)
        rows = [
            __import__("json").loads(line)
            for line in (self.client.folder / "native-ticks.jsonl")
            .read_text()
            .splitlines()
        ]
        self.assertTrue(any(row["kind"] == 2 for row in rows))

    def test_independent_supervision_expires_while_parent_polls(self):
        self.supervisor_stop.set()
        result = self.wait("ABORTED")
        self.assertEqual(result["reason"], "native_supervisor_expired")
        self.assertTrue(self.client.process.poll() is None)


class NativeJournalFailureTests(unittest.TestCase):
    def test_native_quota_failure_is_not_a_complete_report(self):
        with tempfile.TemporaryDirectory() as root:
            client = NativeClient(LoopConfig(max_log_mb=1), root, simulated=True)
            try:
                client.command(
                    "bind",
                    q=[0.0] * 14,
                    bounds=[[-2.0, 2.0]] * 14,
                    boot="test",
                    session=str(uuid.uuid4()),
                )
                client.command("activate")
                deadline = time.monotonic() + 2
                while time.monotonic() < deadline:
                    client.supervise(time.monotonic())
                    status = client.status()
                    if status["recording_failed"]:
                        break
                    time.sleep(0.01)
                self.assertTrue(status["recording_failed"])
                with self.assertRaisesRegex(ControlFault, "native_journal_failure"):
                    client.close()
            finally:
                if client.process.poll() is None:
                    client.process.terminate()
                    client.process.wait(timeout=3)


if __name__ == "__main__":
    unittest.main()
