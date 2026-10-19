"""Controller boundary and read-only hardware readiness evidence.

The legacy TCP HardwareBackend stays disabled. Stage B joint execution is
implemented separately in joint_execution.py and ros_execution_transport.py;
production transport is stationary-only and is not exposed by the web server.
"""

import json
import time
import urllib.request
import uuid
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from robot_control.command_bridge import Feedback, ControlFault, current_action
from robot_control.common import sha256
from robot_control.action_adapter import prepare_actions


class HardwareBackend:
    capabilities = {
        "atomic_ownership": False,
        "bounded_submit": False,
        "stop_receipts": False,
        "independent_watchdog": False,
    }

    def acquire(self, now):
        raise ControlFault(
            "hardware_transport_unavailable: ownership/stop/watchdog contract not implemented or verified"
        )

    def read(self, now):
        raise ControlFault("hardware_transport_unavailable")

    def write(self, *args):
        raise ControlFault("hardware_transport_unavailable")

    def stop_channel(self, *args):
        return False

    def release(self, *args):
        raise ControlFault("hardware_transport_unavailable")

    def close(self):
        pass


def encode_targets(action, joints, stamp_ns):
    """Pure conversion to existing ROS messages. Does not create a node/publisher."""
    from marvin_msgs.msg import IKRequest
    from std_msgs.msg import Float32

    action = np.asarray(action, dtype=np.float64)
    joints = np.asarray(joints, dtype=np.float64)
    if (
        action.shape != (14,)
        or joints.shape != (14,)
        or not np.isfinite(action).all()
        or not np.isfinite(joints).all()
    ):
        raise ValueError("Invalid target arrays")
    if np.any(action[[6, 13]] < 0) or np.any(action[[6, 13]] > 1):
        raise ValueError("Invalid gripper target")
    if type(stamp_ns) is not int or stamp_ns <= 0:
        raise ValueError("Invalid stamp")
    message = IKRequest()
    message.header.frame_id = "base_link"
    message.header.stamp.sec = stamp_ns // 10**9
    message.header.stamp.nanosec = stamp_ns % 10**9
    message.available = [True, True] + [False] * 9
    for offset, name in [(0, "left_hand_pose"), (7, "right_hand_pose")]:
        pose = getattr(message, name)
        pose.position.x, pose.position.y, pose.position.z = map(
            float, action[offset : offset + 3]
        )
        q = Rotation.from_euler("xyz", action[offset + 3 : offset + 6]).as_quat()
        (
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        ) = map(float, q)
    message.joint_state.header = message.header
    message.joint_state.name = [
        f"Joint{i}_{s}" for s in ["L", "R"] for i in range(1, 8)
    ]
    message.joint_state.position = joints.tolist()
    left, right = Float32(), Float32()
    left.data = float(action[6])
    right.data = float(action[13])
    return message, left, right


class SimulatedBackend:
    """Idealized feedback/lease simulator, not robot dynamics or QP simulation."""

    def __init__(self, state, watchdog_s=0.1, fault=None):
        self.state = np.asarray(state, dtype=float).copy()
        current_action(self.state)
        self.watchdog_s = watchdog_s
        self.fault = fault
        self.token = None
        self.last_write = None
        self.events = []
        self.closed = False
        self.stop_calls = []
        self.write_count = 0
        self.hardware_writes = 0

    def acquire(self, now):
        if self.token or self.fault == "ownership_busy":
            raise ControlFault("ownership_busy")
        self.token = uuid.uuid4().hex
        self.last_write = now
        return self.token

    def watchdog(self, now):
        if self.token and now - self.last_write > self.watchdog_s:
            self.events.append({"event": "independent_watchdog", "time": now})
            self.token = None

    def read(self, now):
        self.watchdog(now)
        if self.fault == "disconnect":
            raise ConnectionError("simulated_disconnect")
        received = now - 1 if self.fault == "stale_feedback" else now
        owner = "other" if self.fault == "owner_lost" else self.token
        gripper_owner = "other" if self.fault == "gripper_owner_lost" else self.token
        state = self.state.copy()
        if self.fault == "tracking_error":
            state[0] += 0.1
        if self.fault == "nan_feedback":
            state[0] = np.nan
        return Feedback(
            state, received, owner, gripper_owner, True, self.fault != "ik_failure"
        )

    def write(self, action, token, now):
        if token != self.token:
            raise ControlFault("invalid_lease")
        self.last_write = now
        self.write_count += 1
        self.state[:7] = action[:7]
        if self.fault == "right_write_failure":
            raise ConnectionError("right_write_failed_after_left")
        self.state[13:20] = action[7:]
        self.events.append({"event": "write", "time": now, "target": action.tolist()})
        return {"left": True, "right": True, "grippers": self.fault != "gripper_reject"}

    def stop_channel(self, channel, token, now):
        self.stop_calls.append(channel)
        return self.fault != "stop_unconfirmed"

    def release(self, token, now):
        if self.fault == "release_failure":
            raise ConnectionError("release_failed")
        self.token = None

    def close(self):
        self.closed = True


def audit_prediction(folder):
    folder = Path(folder)
    reasons = []
    report = json.loads((folder / "report.json").read_text())
    for name, digest in report.get("artifacts", {}).items():
        path = (folder / name).resolve()
        if (
            path.parent != folder.resolve()
            or not path.is_file()
            or sha256(path) != digest
        ):
            reasons.append("artifact_hash_mismatch:" + name)
    with np.load(folder / "prediction.npz", allow_pickle=False) as data:
        actions = data["actions"].copy()
    if actions.shape != (1, 50, 14) or not np.isfinite(actions).all():
        reasons.append("invalid_prediction")
    elif (folder / "action_adaptation.json").exists():
        adaptation = json.loads((folder / "action_adaptation.json").read_text())
        allowed = adaptation.get("max_allowed_correction")
        if (
            type(allowed) not in (int, float)
            or not np.isfinite(allowed)
            or not 0 <= allowed <= 0.05
        ):
            reasons.append("unverified_gripper_correction_budget")
        else:
            expected, audit = prepare_actions(actions, allowed)
            if adaptation != audit:
                reasons.append("action_adaptation_mismatch")
            if expected is None:
                reasons.extend(audit["reasons"])
            elif not (folder / "execution_candidate.npz").exists():
                reasons.append("execution_candidate_missing")
            else:
                required = {
                    "prediction.npz",
                    "action_adaptation.json",
                    "execution_candidate.npz",
                }
                if not required.issubset(report.get("artifacts", {})):
                    reasons.append("candidate_artifact_hashes_missing")
                with np.load(
                    folder / "execution_candidate.npz", allow_pickle=False
                ) as candidate:
                    if not np.array_equal(candidate["actions"], expected):
                        reasons.append("execution_candidate_mismatch")
    elif np.any(actions[0, :, [6, 13]] < 0) or np.any(actions[0, :, [6, 13]] > 1):
        reasons.append("gripper_out_of_range")
    reasons.extend(report.get("action_reasons", []))
    reasons.extend(report.get("ik_reasons", []))
    reasons.extend(report.get("post_analysis_reasons", []))
    if report.get("status") != "accepted_by_diagnostic_gates":
        reasons.append("prediction_not_accepted")
    snapshot = report.get("snapshot", {})
    age = (time.time_ns() - snapshot.get("snapshot_unix_ns", 0)) / 1e9
    if not 0 <= age <= 2.5:
        reasons.append("saved_prediction_stale_not_replayable")
    ik = json.loads((folder / "ik_checks.json").read_text())
    if not ik.get("environment_collision_checked"):
        reasons.append("environment_collision_unverified")
    if not ik.get("manifest", {}).get("online_qp_equivalence_verified"):
        reasons.append("online_qp_equivalence_unverified")
    return {
        "reasons": sorted(set(reasons)),
        "prediction_age_s": age,
        "source_report_sha256": sha256(folder / "report.json"),
    }


def inspect_control_graph(domain):
    import rclpy
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.signals import SignalHandlerOptions

    context = Context()
    node = None
    executor = None
    try:
        rclpy.init(
            args=[],
            context=context,
            domain_id=domain,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        node = Node(
            "forcevla_control_readiness_only",
            context=context,
            enable_rosout=False,
            start_parameter_services=False,
            enable_logger_service=False,
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
        result = {}
        for topic in [
            "/tj/control/ik_request",
            "/tj/control/joint_cmd_A",
            "/tj/control/joint_cmd_B",
            "/control/gripperValueL",
            "/control/gripperValueR",
        ]:
            result[topic] = {
                "publishers": [
                    p.node_namespace.rstrip("/") + "/" + p.node_name
                    for p in node.get_publishers_info_by_topic(topic)
                ],
                "subscribers": [
                    p.node_namespace.rstrip("/") + "/" + p.node_name
                    for p in node.get_subscriptions_info_by_topic(topic)
                ],
            }
        return result
    finally:
        if executor:
            executor.shutdown(timeout_sec=1)
        if node:
            node.destroy_node()
        if context.ok():
            context.shutdown()


def gripper_claim_readiness(response, max_feedback_age_s, motor_velocity_rad_s=None):
    """Inspect claim prerequisites, using the wrapper's existing-target seed.

    WebGripper.command replaces measured with normalized q1/q2 during claim.
    Velocity and tracking error are not claim prerequisites. This read-only
    audit never sends a claim or confirms ownership.
    """
    reasons = []
    position_error = None
    if response.get("ok") is not True or not isinstance(response.get("data"), dict):
        return {
            "driver_claim_preconditions_passed": False,
            "reasons": ["gripper_status_unavailable"],
            "claim_request_sent": False,
        }
    data = response["data"]
    if data.get("reserved") is not False:
        reasons.append("gripper_already_reserved_or_unknown")
    if data.get("healthy") is not True:
        reasons.append("gripper_unhealthy")
    try:
        actual = np.asarray(data.get("actual"), dtype=float)
        target = np.asarray(data.get("applied_target"), dtype=float)
        calibration = np.asarray(data.get("calibration"), dtype=float)
        if (
            actual.shape != (2,)
            or target.shape != (2,)
            or calibration.shape != (2, 2)
            or not np.isfinite(actual).all()
            or not np.isfinite(target).all()
            or not np.isfinite(calibration).all()
            or np.any(calibration[:, 1] <= calibration[:, 0])
        ):
            raise ValueError("invalid status vectors")
        if np.any(target < 0) or np.any(target > 1):
            reasons.append("gripper_existing_target_outside_claim_range")
        position_error = np.abs(actual - target) * (
            calibration[:, 1] - calibration[:, 0]
        )
    except (ValueError, TypeError):
        reasons.append("gripper_claim_status_invalid")
    driver_ready = not reasons
    try:
        ages = np.asarray(data.get("motor_feedback_age"), dtype=float)
        status_age = float(data.get("status_age_sec", float("nan")))
        if (
            ages.shape != (2,)
            or not np.isfinite(ages).all()
            or np.any(ages < 0)
            or not np.isfinite(status_age)
            or status_age < 0
            or np.any(ages + status_age > max_feedback_age_s)
        ):
            reasons.append("gripper_feedback_stale_or_unknown")
    except (ValueError, TypeError):
        reasons.append("gripper_feedback_stale_or_unknown")
    return {
        "driver_claim_preconditions_passed": driver_ready,
        "freshness_checked_separately": True,
        "actual": data.get("actual"),
        "claim_seed_existing_target": data.get("applied_target"),
        "position_error_rad": position_error.tolist()
        if position_error is not None
        else None,
        "reasons": reasons,
        "claim_request_sent": False,
        "claim_semantics": "wrapper seeds existing q1/q2 target; no velocity or tracking-error gate; health, ownership and target range required",
    }


def readiness_audit(config, prediction_folder=None, live=True):
    reasons = [
        "hardware_transport_unavailable",
        "exclusive_ik_ownership_unverified",
        "combined_arm_gripper_stop_unverified",
        "hardware_watchdog_unverified",
        "site_motion_limits_unverified",
    ]
    for key in [
        "camera_mapping_verified",
        "camera_transport_equivalence_verified",
        "physical_tcp_verified",
        "wrench_physical_semantics_verified",
    ]:
        if not config.get(key):
            reasons.append(key + ":false")
    out = {
        "status": "blocked_hardware_execution",
        "hardware_commands_sent": 0,
        "controller_capabilities": HardwareBackend.capabilities,
        "stage_b_backend": {
            "implementation": "joint_execution.JointExecutionBackend",
            "transport": "ros_execution_transport.RosExecutionTransport",
            "production_scope": "stationary_only_not_deployed",
            "hardware_acceptance_passed": False,
            "moving_scheduler_available": False,
        },
        "live_status": {},
    }
    if prediction_folder:
        out["prediction"] = audit_prediction(prediction_folder)
        reasons.extend(out["prediction"]["reasons"])
    else:
        reasons.append("no_accepted_fresh_prediction")
    if live:
        for key, url in config["http_status"].items():
            try:
                with urllib.request.urlopen(url, timeout=3) as response:
                    out["live_status"][key] = json.load(response)
            except Exception as exc:
                out["live_status"][key] = {"error": str(exc)}
                reasons.append("status_unavailable:" + key)
    if live:
        out["gripper_claim_readiness"] = gripper_claim_readiness(
            out["live_status"].get("gripper", {}),
            config["limits"]["max_motor_feedback_age_s"],
        )
        reasons.extend(out["gripper_claim_readiness"]["reasons"])
        try:
            out["control_graph"] = inspect_control_graph(config["ros_domain_id"])
            if out["control_graph"]["/tj/control/ik_request"]["publishers"]:
                reasons.append("ik_input_has_existing_publishers_no_exclusive_lease")
        except Exception as exc:
            out["control_graph_error"] = repr(exc)
            reasons.append("control_graph_unavailable")
    out["blocking_reasons"] = sorted(set(reasons))
    return out
