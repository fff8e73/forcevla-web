"""Concrete ROS transport for Stage B; production targets are stationary-only.

Importing this module never initializes ROS. The isolated runner fixes domain231
and a random namespace. Production construction is explicit and not exposed by
run_control or the web server. No robot SDK, shared IK or direct driver writes.
"""

import copy
import json
import threading
import time
import uuid

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import ExecutionFeedback


class RosExecutionTransport:
    def __init__(self, *, domain, namespace, timeout_s=0.4, allow_missing_targets=False, create_arm_publishers=True):
        if not (
            domain == 24
            and namespace == "/tj"
            or domain == 231
            and namespace.startswith("/forcevla_sandbox_")
        ):
            raise ValueError(
                "Only explicit production or isolated sandbox endpoints allowed"
            )
        if not 0 < timeout_s <= 1:
            raise ValueError("Invalid IO deadline")
        self.allow_missing_targets = allow_missing_targets
        self.production = domain == 24
        self.namespace, self.timeout = namespace, timeout_s
        self.seed = self.grip_seed = None
        self.lock = threading.RLock()
        self.changed = threading.Condition(self.lock)
        self.samples = {}
        self.closed = False
        self.failure = None
        self.publications = 0
        self.operation_counts = dict(input_requests=0, left=0, right=0, gripper_commands=0, gripper_heartbeats=0)
        import rclpy
        from marvin_msgs.msg import JointcmdArm, Jointfeedback
        from marvin_msgs.srv import Int
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import Float32MultiArray, Int16MultiArray, Int32, String

        self.JointcmdArm, self.String, self.Int = JointcmdArm, String, Int
        self.context = Context()
        rclpy.init(
            args=[],
            context=self.context,
            domain_id=domain,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        self.node = Node(
            "forcevla_executor_" + uuid.uuid4().hex[:8],
            namespace=namespace,
            context=self.context,
            enable_rosout=False,
            start_parameter_services=False,
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.qos = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        sensor = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        self.arms = {
            side: self.node.create_publisher(
                JointcmdArm, namespace + "/control/user/joint_cmd_" + letter, self.qos
            )
            for side, letter in [("left", "A"), ("right", "B")]
        } if create_arm_publishers else {}
        # Tool topics are global on production; every sandbox topic is namespaced.
        tool = "" if self.production else namespace
        self.grip_pub = self.node.create_publisher(
            String, tool + "/control/web_gripper", self.qos
        )
        self.beat_pub = self.node.create_publisher(
            String, tool + "/control/web_gripper/heartbeat", self.qos
        )
        self.selector = self.node.create_client(Int, namespace + "/control/set_input")
        specs = [
            (Jointfeedback, namespace + "/info/joint_feedback", "joints", sensor),
            (Int32, namespace + "/control/input_mode", "input", latched),
            (Int16MultiArray, namespace + "/info/arm_state", "mode", sensor),
            (String, tool + "/info/web_gripper", "gripper", sensor),
            (
                Float32MultiArray,
                tool + "/info/gripper_feedback_L",
                "motor_left",
                sensor,
            ),
            (
                Float32MultiArray,
                tool + "/info/gripper_feedback_R",
                "motor_right",
                sensor,
            ),
        ]
        specs += [
            (
                JointcmdArm,
                namespace + "/control/joint_cmd_" + letter,
                "target_" + side,
                sensor,
            )
            for side, letter in [("left", "A"), ("right", "B")]
        ]
        self.subs = [
            self.node.create_subscription(kind, topic, self.callback(key), qos)
            for kind, topic, key, qos in specs
        ]
        self.thread = threading.Thread(
            target=self.spin, name="execution-ros", daemon=True
        )
        self.thread.start()

    def spin(self):
        try:
            self.executor.spin()
        except Exception as exc:  # noqa: BLE001 - preserve transport failures in report
            self.failure = repr(exc)
            with self.changed:
                self.changed.notify_all()

    def callback(self, key):
        def receive(message):
            with self.changed:
                self.samples[key] = (message, time.monotonic())
                self.changed.notify_all()

        return receive

    def sample(self, key):
        if self.closed or self.failure:
            raise ControlFault("ROS_transport_closed_or_failed:" + str(self.failure))
        if key not in self.samples:
            raise ControlFault("missing_ROS_feedback:" + key)
        return self.samples[key]

    def gripper_status(self):
        with self.lock:
            msg, received = self.sample("gripper")
            if time.monotonic() - received > 0.3:
                raise ControlFault("gripper_status_stale")
            g = json.loads(msg.data)
            source_age = (time.time_ns() - g.get("sample_time_ns", 0)) / 1e9
            if not 0 <= source_age <= 0.3:
                raise ControlFault("gripper_source_stamp_stale")
            # Include transport/source delay in the driver's motor ages.
            g["motor_feedback_age"] = [
                float(x) + max(0, source_age - (time.monotonic() - received))
                for x in g["motor_feedback_age"]
            ]
            v = []
            for side in ("left", "right"):
                motor, at = self.sample("motor_" + side)
                if time.monotonic() - at > 0.3 or len(motor.data) < 2:
                    raise ControlFault("motor_velocity_stale")
                v.append(float(motor.data[1]))
            g["motor_velocities"] = v
            return g

    def feedback(self):
        with self.lock:
            joint, received = self.sample("joints")
            selected, _ = self.sample("input")
            mode, mode_at = self.sample("mode")
            if self.allow_missing_targets and any('target_'+s not in self.samples for s in ('left','right')):
                target_q, target_time = np.full(14, np.nan), 0.0
            else:
                targets = [self.sample("target_" + s) for s in ("left", "right")]
                target_q = np.concatenate([np.array(t[0].positions) for t in targets])
                target_time = min(t[1] for t in targets)
            g = self.gripper_status()
            _, g_at = self.sample("gripper")
            stamp = joint.header.stamp.sec + joint.header.stamp.nanosec / 1e9
            modes = list(mode.data)
            healthy = (
                time.monotonic() - mode_at <= 0.3
                and len(modes) >= 2
                and modes[0] == modes[1]
                and modes[0] in (1, 3)
            )
            unexpected = False
            for pub in self.arms.values():
                infos = self.node.get_publishers_info_by_topic(pub.topic_name)
                if len(infos) != 1 or any(
                    p.node_name != self.node.get_name()
                    or p.node_namespace != self.namespace
                    for p in infos
                ):
                    unexpected = True
            return ExecutionFeedback(
                received,
                time.time() - stamp,
                np.array(joint.arm_positions),
                np.array(joint.arm_velocities),
                target_q,
                target_time,
                int(selected.data),
                healthy,
                modes[0] if modes else -1,
                copy.deepcopy(g),
                g_at,
                unexpected,
            )

    def wait(self, predicate, label):
        end = time.monotonic() + self.timeout
        with self.changed:
            while True:
                if self.failure or self.closed:
                    raise ControlFault("transport_unavailable")
                value = predicate()
                if value is not None:
                    return value
                remaining = end - time.monotonic()
                if remaining <= 0:
                    raise ControlFault(label + "_timeout_no_retry")
                self.changed.wait(min(remaining, 0.02))

    def select_input(self, index):
        if index not in (0, 3):
            raise ControlFault("invalid_input_selection")
        if not self.selector.service_is_ready():
            raise ControlFault("input_service_unavailable")
        started = time.monotonic()
        with self.lock:
            prior = self.samples.get("input")
            already_selected = prior is not None and prior[0].data == index
        req = self.Int.Request()
        req.data = index
        self.operation_counts['input_requests'] += 1
        future = self.selector.call_async(req)
        result = self.wait(
            lambda: future.result() if future.done() else None, "input_service"
        )
        if not result.success:
            raise ControlFault("input_service_rejected")

        def readback():
            item = self.samples.get("input")
            return (
                True
                if item
                and (already_selected or item[1] >= started)
                and item[0].data == index
                else None
            )

        self.wait(readback, "input_readback")

    def bind_stationary_target(self, joints, grippers):
        if self.seed is not None:
            raise ControlFault("transport_session_cannot_rebind")
        self.seed = np.array(joints, copy=True)
        self.grip_seed = np.array(grippers, copy=True)

    def validate_targets(self, joints, grippers):
        if self.seed is None:
            raise ControlFault("transport_not_bound")
        if self.production and (
            not np.array_equal(joints, self.seed)
            or not np.array_equal(grippers, self.grip_seed)
        ):
            raise ControlFault("production_stage_B_stationary_only")

    def publish_arm(self, side, positions):
        q = np.asarray(positions, float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ControlFault("invalid_arm_command")
        index = 0 if side == "left" else 7
        if self.production and (
            self.seed is None or not np.array_equal(q, self.seed[index : index + 7])
        ):
            raise ControlFault("production_stage_B_stationary_only")
        pub = self.arms[side]
        if pub.get_subscription_count() < 1:
            raise ControlFault("arm_mux_subscriber_unavailable:" + side)
        msg = self.JointcmdArm()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.positions = q.tolist()
        pub.publish(msg)
        self.operation_counts[side] += 1
        self.publications += 1

    def gripper_command(self, request):
        if request.get("action") not in ("claim", "target", "hold", "release"):
            raise ControlFault("invalid_gripper_action")
        if self.production and request["action"] == "target":
            raise ControlFault("production_stage_B_no_gripper_target")
        if self.grip_pub.get_subscription_count() < 1:
            raise ControlFault("gripper_subscriber_unavailable")
        msg = self.String()
        msg.data = json.dumps(request, allow_nan=False)
        self.operation_counts['gripper_commands'] += 1
        self.grip_pub.publish(msg)
        self.publications += 1

        def receipt():
            if "gripper" not in self.samples:
                return None
            g = self.gripper_status()
            return g if g.get("last_request_id") == request["request_id"] else None

        return self.wait(receipt, "gripper_command")

    def heartbeat(self, request):
        if self.beat_pub.get_subscription_count() < 1:
            raise ControlFault("gripper_heartbeat_subscriber_unavailable")
        msg = self.String()
        msg.data = json.dumps(request, allow_nan=False)
        self.operation_counts['gripper_heartbeats'] += 1
        self.beat_pub.publish(msg)

    def close(self):
        if self.closed:
            return
        self.closed = True
        with self.changed:
            self.changed.notify_all()
        self.executor.shutdown(timeout_sec=1)
        self.thread.join(timeout=1)
        self.node.destroy_node()
        self.context.shutdown()
