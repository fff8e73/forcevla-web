import sys
from pathlib import Path
import unittest
import numpy as np

from robot_control.joint_source_guard import JointEnvelope, JointSourceGuard, GateRejected


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.g = JointSourceGuard(
            np.tile([-2.0, 2.0], (14, 1)), np.ones(14), lease_s=0.2
        )
        self.token = self.g.claim("test", 10.0, np.zeros(14), 10.0, 3)

    def command(
        self, sequence=0, command_time=10.01, observation_time=10.0, q=None, now=10.01
    ):
        return self.g.accept(
            JointEnvelope(
                self.token,
                sequence,
                command_time,
                observation_time,
                np.zeros(14) if q is None else q,
            ),
            now,
            3,
            feedback_time=now,
        )

    def test_valid_copy_readonly(self):
        q = np.full(14, 0.001)
        out = self.command(q=q)
        q[:] = 1
        np.testing.assert_array_equal(out, np.full(14, 0.001))
        self.assertFalse(out.flags.writeable)

    def test_wrong_client_cannot_take_over_or_trip_owner(self):
        with self.assertRaises(GateRejected):
            self.g.claim("other", 10.01, np.zeros(14), 10.01, 3)
        with self.assertRaises(GateRejected):
            self.g.heartbeat("wrong", 10.01, 3)
        self.assertEqual(self.g.state, "ACTIVE")
        self.command()

    def test_old_header_is_rejected_even_if_mux_would_accept(self):
        with self.assertRaisesRegex(GateRejected, "command_stale"):
            self.command(command_time=9.0, observation_time=8.0)
        self.assertEqual(self.g.state, "TRIPPED")

    def test_duplicate_and_reordered_sequence(self):
        self.command()
        with self.assertRaisesRegex(GateRejected, "sequence"):
            self.command(command_time=10.02, now=10.02)

    def test_heartbeat_cannot_revive_old_observation(self):
        self.g.heartbeat(self.token, 10.1, 3)
        with self.assertRaisesRegex(GateRejected, "prediction_stale"):
            self.command(command_time=10.1, now=10.1, observation_time=7.0)

    def test_commands_do_not_renew_lease(self):
        for i in range(1, 10):
            self.command(sequence=i, command_time=10 + i * 0.02, now=10 + i * 0.02)
        result = self.g.poll(10.21)
        self.assertEqual(result["state"], "TRIPPED")
        self.assertTrue(result["stop_required"])
        with self.assertRaises(GateRejected):
            self.g.heartbeat(self.token, 10.22, 3)

    def test_mode_change_and_external_publisher_trip(self):
        with self.assertRaisesRegex(GateRejected, "selection"):
            self.g.heartbeat(self.token, 10.01, 1)
        self.g.finish(self.token, independent_stop_confirmed=True)
        token = self.g.claim("new", 11, np.zeros(14), 11, 3)
        with self.assertRaisesRegex(GateRejected, "publisher"):
            self.g.heartbeat(token, 11.01, 3, True)

    def test_stop_confirmation_and_generation(self):
        old = self.token
        self.g.poll(10.3)
        with self.assertRaisesRegex(GateRejected, "not_confirmed"):
            self.g.finish(old, independent_stop_confirmed=False)
        self.g.finish(old, independent_stop_confirmed=True)
        token = self.g.claim("new", 11, np.zeros(14), 11, 3)
        self.assertNotEqual(old, token)
        with self.assertRaises(GateRejected):
            self.g.heartbeat(old, 11.01, 3)

    def test_joint_speed_and_step(self):
        with self.assertRaisesRegex(GateRejected, "velocity"):
            self.command(q=np.full(14, 0.02))
        self.g.finish(self.token, independent_stop_confirmed=True)
        self.token = self.g.claim("new", 11, np.zeros(14), 11, 3)
        self.g.heartbeat(self.token, 11.1, 3)
        with self.assertRaisesRegex(GateRejected, "step"):
            self.command(
                command_time=11.1, observation_time=11, now=11.1, q=np.full(14, 0.08)
            )

    def test_joint_limit_and_stale_feedback_are_latched(self):
        with self.assertRaisesRegex(GateRejected, "joint_limit"):
            self.command(q=np.full(14, 3.0))
        self.g.finish(self.token, independent_stop_confirmed=True)
        self.token = self.g.claim("new", 11, np.zeros(14), 11, 3)
        with self.assertRaisesRegex(GateRejected, "feedback_stale"):
            self.g.accept(
                JointEnvelope(self.token, 0, 11.01, 11, np.zeros(14)),
                11.01,
                3,
                feedback_time=10.0,
            )
        self.assertEqual(self.g.state, "TRIPPED")

    def test_claim_requires_input_3_and_fresh_feedback(self):
        gate = JointSourceGuard(np.tile([-2.0, 2.0], (14, 1)), np.ones(14))
        with self.assertRaisesRegex(GateRejected, "not_selected"):
            gate.claim("x", 10, np.zeros(14), 10, 1)
        with self.assertRaisesRegex(GateRejected, "feedback_stale"):
            gate.claim("x", 10, np.zeros(14), 9, 3)
        self.assertEqual(gate.state, "IDLE")

    def test_bounds_nan_clock_and_feedback(self):
        with self.assertRaisesRegex(GateRejected, "invalid_joint"):
            self.command(q=np.full(14, np.nan))
        self.g.finish(self.token, independent_stop_confirmed=True)
        self.token = self.g.claim("new", 11, np.zeros(14), 11, 3)
        with self.assertRaisesRegex(GateRejected, "clock"):
            self.g.heartbeat(self.token, 10.9, 3)


if __name__ == "__main__":
    unittest.main()
