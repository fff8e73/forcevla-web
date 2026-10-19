"""Live adapters for V1. Qualification is checked before ROS construction.

This is a distinct plan-authorized transport. Stage B's stationary adapter is
unchanged; changing its production flag is neither used nor supported.
"""

import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np

from robot_control.closed_loop import ClosedLoop
from robot_control.command_bridge import ControlFault
from robot_control.ros_execution_transport import RosExecutionTransport

DEPLOYMENT_MANIFEST = Path(
    "/home/user/forcevla-runtime-thor/artifacts/deployed-runtime.json"
)


def runtime_identity():
    """Freeze the deployed model declaration and verify every numerical asset."""
    planning = Path(__file__).resolve().parents[1] / "config/planning.json"
    limits = json.loads(planning.read_text())
    assets = {str(planning): hashlib.sha256(planning.read_bytes()).hexdigest()}
    for path, expected in limits["solver_file_sha256"].items():
        actual = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        if actual != expected:
            raise ControlFault("runtime_asset_changed:" + path)
        assets[path] = actual
    deployment = DEPLOYMENT_MANIFEST.read_bytes()
    return dict(
        deployment_sha256=hashlib.sha256(deployment).hexdigest(),
        deployment_manifest=json.loads(deployment),
        asset_sha256=assets,
    )


def check_qualification(path, config):
    if not path or not Path(path).is_file():
        raise ControlFault("motion_qualification_missing_M6_required")
    qualification = json.loads(Path(path).read_text())
    if (
        qualification.get("schema_version") != 1
        or qualification.get("config_sha256") != config.digest()
    ):
        raise ControlFault("qualification_profile_mismatch")
    if qualification.get("source_sha256") != ClosedLoop._source_hashes():
        raise ControlFault("qualification_code_mismatch")
    if (
        not time.time()
        < qualification.get("expires_unix", 0)
        <= time.time() + 7 * 86400
    ):
        raise ControlFault("qualification_expired_or_invalid")
    identity = runtime_identity()
    if qualification.get("deployment_sha256") != identity["deployment_sha256"]:
        raise ControlFault("qualification_model_mismatch")
    if qualification.get("asset_sha256") != identity["asset_sha256"]:
        raise ControlFault("qualification_assets_mismatch")
    checks = {
        "finite_motion": ("actual_tracking_passed", "start_stop_bounds_passed"),
        "moving_stop": (
            "manual_stop_passed",
            "upstream_loss_stop_passed",
            "process_loss_stop_passed",
        ),
        "timing": ("submission_deadlines_passed", "loaded_timing_passed"),
        "site": ("workspace_checked", "tool_frames_checked", "gripper_motion_checked"),
    }
    verified = {}
    for kind, required in checks.items():
        entry = qualification.get("reports", {}).get(kind, {})
        report_path = Path(entry.get("path", ""))
        if not report_path.is_file():
            raise ControlFault("qualification_report_missing:" + kind)
        content = report_path.read_bytes()
        if hashlib.sha256(content).hexdigest() != entry.get("sha256"):
            raise ControlFault("qualification_report_hash:" + kind)
        report = json.loads(content)
        if (
            report.get("mode") != "live"
            or report.get("synthetic") is not False
            or report.get("config_sha256") != config.digest()
            or report.get("source_sha256") != qualification["source_sha256"]
            or report.get("deployment_sha256") != identity["deployment_sha256"]
            or report.get("asset_sha256") != identity["asset_sha256"]
            or any(report.get(k) is not True for k in required)
        ):
            raise ControlFault("qualification_report_not_passed:" + kind)
        if (
            not isinstance(report.get("measurements"), dict)
            or not report["measurements"]
        ):
            raise ControlFault("qualification_measurements_missing:" + kind)
        measurements = report["measurements"]
        thresholds = {
            "finite_motion": {
                "max_joint_speed_rad_s": config.max_joint_speed_rad_s,
                "max_joint_acceleration_rad_s2": config.max_joint_acceleration_rad_s2,
                "max_tracking_error_rad": config.tracking_error_rad,
            },
            "moving_stop": {
                "max_stop_latency_s": config.hold_timeout_s,
                "max_final_joint_speed_rad_s": config.stop_speed_rad_s,
            },
            "timing": {"max_submission_lateness_s": config.max_lateness_s},
            "site": {
                "tool_frame_error_m": config.max_tcp_tracking_m,
                "tool_frame_error_rad": config.max_tcp_tracking_rad,
            },
        }[kind]
        for key, upper in thresholds.items():
            value = measurements.get(key)
            if (
                type(value) not in (int, float)
                or not np.isfinite(value)
                or not 0 <= value <= upper
            ):
                raise ControlFault(
                    "qualification_measurement_out_of_budget:" + kind + ":" + key
                )
        count = measurements.get("samples")
        if type(count) is not int or count < (150 if kind == "timing" else 3):
            raise ControlFault("qualification_measurement_samples:" + kind)
        verified[kind] = dict(path=str(report_path), sha256=entry["sha256"])
    return dict(config_sha256=config.digest(), reports=verified)


class RosMotionTransport(RosExecutionTransport):
    bounded_cached_feedback = True
    has_source_ramp = True

    def __init__(
        self,
        config,
        qualification_path=None,
        *,
        domain=24,
        namespace="/tj",
        create_arm_publishers=True,
    ):
        if domain == 24 and namespace == "/tj":
            self.qualification = check_qualification(qualification_path, config)
        elif domain == 231 and namespace.startswith("/forcevla_sandbox_"):
            self.qualification = dict(
                config_sha256=config.digest(), scope="isolated_ROS_only"
            )
        else:
            raise ValueError("Explicit production or isolated namespace required")
        self.config = config
        self.cache_lock = threading.Lock()
        self.cached = None
        self.cached_at = 0.0
        self.cache_error = None
        self.subscriber_counts = {}
        self.cache_end = threading.Event()
        self.motion_lock = threading.RLock()
        self.approved = None
        self.allowed_rows = set()
        self.allowed_grips = [set(), set()]
        self.pending_pair = None
        self.pair_side = None
        self.terminal = False
        super().__init__(
            domain=domain,
            namespace=namespace,
            timeout_s=config.gripper_receipt_timeout_s,
            allow_missing_targets=True,
            create_arm_publishers=create_arm_publishers,
        )
        self.cache_thread = threading.Thread(
            target=self._cache, name="motion-feedback-cache", daemon=True
        )
        self.cache_thread.start()

    def _cache(self):
        while not self.cache_end.is_set():
            try:
                feedback = super().feedback()
                counts = {
                    side: pub.get_subscription_count()
                    for side, pub in self.arms.items()
                }
                counts["heartbeat"] = self.beat_pub.get_subscription_count()
                for array in (
                    feedback.joints,
                    feedback.velocities,
                    feedback.joint_target,
                ):
                    array.flags.writeable = False
                with self.cache_lock:
                    self.subscriber_counts = counts
                    self.cached, self.cached_at, self.cache_error = (
                        feedback,
                        time.monotonic(),
                        None,
                    )
            except Exception as exc:
                with self.cache_lock:
                    self.cache_error = str(exc)
            self.cache_end.wait(0.01)

    def feedback(self):
        with self.cache_lock:
            if (
                self.cache_error
                or self.cached is None
                or time.monotonic() - self.cached_at > self.config.feedback_age_s
            ):
                raise ControlFault(
                    "cached_feedback_unavailable:" + str(self.cache_error)
                )
            f = self.cached
            # source_age was evaluated when the cache was populated.
            from dataclasses import replace

            return replace(
                f, source_age_s=f.source_age_s + time.monotonic() - self.cached_at
            )

    def select_input(self, index):
        super().select_input(index)
        feedback = super().feedback()
        with self.cache_lock:
            self.cached, self.cached_at, self.cache_error = (
                feedback,
                time.monotonic(),
                None,
            )

    def arm_segment(self, segment):
        with self.motion_lock:
            if self.terminal or self.seed is None:
                raise ControlFault("motion_transport_not_active")
            if (
                segment.checks.get("config_sha256")
                != self.qualification["config_sha256"]
                or segment.checks.get("status") != "prepared"
            ):
                raise ControlFault("unqualified_plan")
            if (
                time.monotonic()
                + segment.duration_s
                + segment.checks["max_brake_s"]
                - segment.observed_at
                > self.config.max_prediction_age_s
            ):
                raise ControlFault("transport_plan_expired")
            self.approved = segment.digest()
            self.allowed_rows = {
                np.ascontiguousarray(q, dtype=float).tobytes()
                for rows in (segment.joints, *segment.brake_paths)
                for q in rows
            }
            self.allowed_grips = [
                set(segment.grippers[:, i]) | {float(segment.initial_grippers[i])}
                for i in range(2)
            ]

    def bind_stationary_target(self, q, grip):
        super().bind_stationary_target(q, grip)
        self.last_applied = np.array(q, copy=True)
        self.allowed_grips = [{float(grip[0])}, {float(grip[1])}]

    def validate_targets(self, joints, grippers):
        q, g = np.asarray(joints, dtype=float), np.asarray(grippers, dtype=float)
        with self.motion_lock:
            if (
                self.seed is None
                or q.shape != (14,)
                or g.shape != (2,)
                or not np.isfinite(q).all()
                or not np.isfinite(g).all()
            ):
                raise ControlFault("invalid_motion_target")
            holding = np.array_equal(q, self.last_applied)
            if not holding and (self.terminal or q.tobytes() not in self.allowed_rows):
                raise ControlFault("target_not_in_checked_plan")
            if any(float(g[i]) not in self.allowed_grips[i] for i in range(2)):
                raise ControlFault("gripper_not_in_checked_plan")
            self.pending_pair = q.copy()
            self.pair_side = "left"

    def publish_arm(self, side, positions):
        q = np.asarray(positions, dtype=float)
        index = 0 if side == "left" else 7
        with self.motion_lock:
            if (
                side != self.pair_side
                or self.pending_pair is None
                or not np.array_equal(q, self.pending_pair[index : index + 7])
            ):
                raise ControlFault("unauthorized_or_reordered_arm_submission")
            pub = self.arms[side]
            if self.subscriber_counts.get(side, 0) < 1:
                raise ControlFault("arm_subscriber_missing:" + side)
            msg = self.JointcmdArm()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.positions = q.tolist()
            pub.publish(msg)
            self.last_applied[index : index + 7] = q
            self.operation_counts[side] += 1
            self.publications += 1
            self.pair_side = "right" if side == "left" else None

    def gripper_command(self, request):
        action = request.get("action")
        if action != "target":
            if action == "hold":
                with self.motion_lock:
                    self.terminal = True
                    self.allowed_rows.clear()
            result = super().gripper_command(request)
            if action == "hold":
                values = np.asarray(result.get("applied_target"), dtype=float)
                if (
                    values.shape != (2,)
                    or not np.isfinite(values).all()
                    or np.any(values < 0)
                    or np.any(values > 1)
                ):
                    raise ControlFault("invalid_hold_gripper_receipt")
                with self.motion_lock:
                    for i in range(2):
                        self.allowed_grips[i].add(float(values[i]))
            return result
        with self.motion_lock:
            side = request.get("side")
            if side not in ("left", "right") or self.terminal or not self.approved:
                raise ControlFault("gripper_motion_not_armed")
            index = 0 if side == "left" else 1
            if request.get("value") not in self.allowed_grips[index]:
                raise ControlFault("gripper_value_not_in_plan")
            if self.grip_pub.get_subscription_count() < 1:
                raise ControlFault("gripper_subscriber_missing")
            msg = self.String()
            msg.data = json.dumps(request, allow_nan=False)
            self.grip_pub.publish(msg)
            self.operation_counts["gripper_commands"] += 1
            self.publications += 1

        def receipt():
            if "gripper" not in self.samples:
                return None
            status = self.gripper_status()
            return (
                status
                if status.get("last_request_id") == request["request_id"]
                else None
            )

        return self.wait(receipt, "gripper_target")

    def heartbeat(self, request):
        if self.subscriber_counts.get("heartbeat", 0) < 1:
            raise ControlFault("heartbeat_subscriber_missing")
        msg = self.String()
        msg.data = json.dumps(request, allow_nan=False)
        self.beat_pub.publish(msg)
        self.operation_counts["gripper_heartbeats"] += 1

    def close(self):
        self.cache_end.set()
        if hasattr(self, "cache_thread"):
            self.cache_thread.join(timeout=1)
        super().close()


def load_compute(config, runtime):
    if runtime.get("model_address") != "127.0.0.1:50071":
        raise ControlFault("V1_requires_the_qualified_local_model_service")
    runtime_identity()
    from robot_control.native_qp import NativeQP

    limits = json.loads(Path(runtime["planning_config"]).read_text())
    # Existing asset hashes remain enforced. Local numerical executable changes
    # are captured by the qualification source manifest as well.
    for path, digest in limits["solver_file_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ControlFault("solver_asset_changed:" + path)
    limits["solver_timeout_s"] = config.prepare_timeout_s
    solver = NativeQP(limits["expected_qp_parameters"], limits)
    sys.path.insert(0, runtime["policy_repo"])
    from examples.forcevla.deployment.policy import ForceVLARemotePolicy
    from examples.forcevla.transport import codec

    client = ForceVLARemotePolicy(
        runtime["model_address"], timeout=config.rpc_timeout_s
    )
    if not client.healthy():
        raise ControlFault("model_unhealthy")

    class Policy:
        def predict(self, raw, request_id):
            request = codec.make_request(
                raw, model_name=client.model_name, request_id=request_id
            )
            timeout = (
                config.warmup_timeout_s
                if request_id.startswith("warmup-")
                else config.rpc_timeout_s
            )
            response = client.stub.Infer(request, timeout=timeout)
            if response.request_id != request_id:
                raise ControlFault("model_response_id_mismatch")
            result = {
                name: codec.decode(value) for name, value in response.outputs.items()
            }
            result["request_id"] = response.request_id
            return result

    return Policy(), solver
