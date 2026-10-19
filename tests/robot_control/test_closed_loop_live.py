import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from robot_control.closed_loop_contracts import LoopConfig
from robot_control.closed_loop_live import RosMotionTransport, check_qualification
from robot_control.command_bridge import ControlFault


class LiveBoundaryTests(unittest.TestCase):
    def test_missing_qualification_fails_before_ros(self):
        with patch("robot_control.ros_execution_transport.RosExecutionTransport.__init__") as ros:
            with self.assertRaisesRegex(ControlFault, "qualification_missing"):
                RosMotionTransport(LoopConfig(), "/not-present")
            ros.assert_not_called()

    def test_boolean_ready_is_not_qualification(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "qualification.json"
            path.write_text(json.dumps(dict(ready=True)))
            with self.assertRaisesRegex(ControlFault, "profile_mismatch"):
                check_qualification(path, LoopConfig())

    def test_arbitrary_ros_domain_never_constructs_transport(self):
        with patch("robot_control.ros_execution_transport.RosExecutionTransport.__init__") as ros:
            with self.assertRaises(ValueError):
                RosMotionTransport(LoopConfig(), domain=1, namespace="/tj")
            ros.assert_not_called()


class QualificationIdentityTests(unittest.TestCase):
    def test_same_code_with_different_model_is_rejected(self):
        import time

        from robot_control.closed_loop import ClosedLoop

        config = LoopConfig()
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "qualification.json"
            path.write_text(
                json.dumps(
                    dict(
                        schema_version=1,
                        config_sha256=config.digest(),
                        source_sha256=ClosedLoop._source_hashes(),
                        expires_unix=time.time() + 3600,
                        deployment_sha256="old-model",
                        asset_sha256={},
                    )
                )
            )
            with patch(
                "robot_control.closed_loop_live.runtime_identity",
                return_value=dict(deployment_sha256="new-model", asset_sha256={}),
            ):
                with self.assertRaisesRegex(ControlFault, "model_mismatch"):
                    check_qualification(path, config)


if __name__ == "__main__":
    unittest.main()
