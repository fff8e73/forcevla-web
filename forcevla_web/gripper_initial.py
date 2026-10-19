"""Send exactly one normalized zero target per unowned gripper. No SDK/mode/reset IO.

Default is read-only; the web operation explicitly passes --execute. A driver
claim is intentionally not requested: the existing external Float32 lane already
rejects writes when the driver is owned. No actuator is accessed on import.
"""

import argparse
import json
import math
import time
import uuid

TOPICS = ("/control/gripperValueL", "/control/gripperValueR")
INITIAL_TARGET = (0.0, 0.0)


def checked_status(io):
    status = io.status()
    if io.input() != 0:
        raise RuntimeError("requires_input_none")
    if status.get("reserved") is not False:
        raise RuntimeError("gripper_owned_do_not_override")
    if status.get("healthy") is not True or status.get("errors") != [1, 1]:
        raise RuntimeError("gripper_not_enabled_or_faulted")
    stamp = status.get("sample_time_ns", 0) / 1e9
    if not math.isfinite(stamp) or not 0 <= time.time() - stamp <= 0.5:
        raise RuntimeError("gripper_source_stale")
    for key in ("actual", "applied_target", "motor_feedback_age"):
        values = status.get(key)
        if (
            not isinstance(values, list)
            or len(values) != 2
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in values)
        ):
            raise RuntimeError("invalid_gripper_feedback:" + key)
    if any(age < 0 or age + time.time() - stamp > 0.5 for age in status["motor_feedback_age"]):
        raise RuntimeError("motor_feedback_stale")
    if status.get("calibration") != [[0.0, 1.6], [0.0, 1.6]]:
        raise RuntimeError("gripper_calibration_changed")
    return status


def run(io, *, execute=False, timeout=6.0, readback_timeout=3.0):
    report = dict(
        status="failed",
        target=list(INITIAL_TARGET),
        target_unit="normalized_0_1",
        topics=list(TOPICS),
        published=[False, False],
        hardware_commands_sent=0,
        goal_readback_confirmed=False,
        physical_arrival_verified=False,
    )
    try:
        deadline = time.monotonic() + timeout
        while True:
            io.spin(0.02)
            if io.discovered():
                break
            if time.monotonic() > deadline:
                raise RuntimeError("gripper_endpoints_unavailable")
        before = checked_status(io)
        report["before"] = before
        if not execute:
            report["status"] = "preflight_passed_no_commands"
            return report
        sent_after = time.time()
        report["sent_after_unix"] = sent_after
        for index in range(2):
            # Check ownership again immediately before each independent send.
            checked_status(io)
            io.publish(index, INITIAL_TARGET[index])
            report["published"][index] = True
            report["hardware_commands_sent"] += 1
        deadline = time.monotonic() + readback_timeout
        while time.monotonic() < deadline:
            io.spin(0.02)
            after = checked_status(io)
            if after["boot_id"] != before["boot_id"]:
                raise RuntimeError("gripper_driver_restarted_during_command")
            if after["sample_time_ns"] / 1e9 > sent_after and all(
                abs(value) <= 1e-6 for value in after["applied_target"]
            ):
                report.update(
                    status="initial_targets_confirmed", goal_readback_confirmed=True, after=after
                )
                return report
        raise RuntimeError("initial_target_readback_timeout_no_retry")
    except Exception as exc:
        report["error"] = str(exc)
        return report


class RosInitialTargetIO:
    def __init__(self):
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import Float32, Int32, String

        self.rclpy, self.Float32 = rclpy, Float32
        self.context = Context()
        rclpy.init(args=[], context=self.context, domain_id=24)
        self.node = Node(
            "forcevla_gripper_initial_" + uuid.uuid4().hex[:8],
            context=self.context,
            enable_rosout=False,
            start_parameter_services=False,
        )
        self.executor = SingleThreadedExecutor(context=self.context)
        self.executor.add_node(self.node)
        self.values = {}
        self.publishers = [self.node.create_publisher(Float32, topic, 1) for topic in TOPICS]
        self.subscriptions = [
            self.node.create_subscription(
                String,
                "/info/web_gripper",
                self.receive_status,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
            ),
            self.node.create_subscription(
                Int32,
                "/tj/control/input_mode",
                self.receive_input,
                QoSProfile(
                    depth=1,
                    reliability=ReliabilityPolicy.RELIABLE,
                    durability=DurabilityPolicy.TRANSIENT_LOCAL,
                ),
            ),
        ]

    def receive_status(self, msg):
        try:
            self.values["gripper"] = (json.loads(msg.data), time.monotonic())
        except (ValueError, TypeError):
            self.values.pop("gripper", None)

    def receive_input(self, msg):
        self.values["input"] = int(msg.data)

    def discovered(self):
        return (
            "input" in self.values
            and "gripper" in self.values
            and all(p.get_subscription_count() > 0 for p in self.publishers)
        )

    def status(self):
        data, received = self.values["gripper"]
        if time.monotonic() - received > 0.5:
            raise RuntimeError("gripper_status_stale")
        return data

    def input(self):
        return self.values["input"]

    def publish(self, index, value):
        if value != 0.0 or index not in (0, 1):
            raise ValueError("only_explicit_initial_zero_targets_allowed")
        self.publishers[index].publish(self.Float32(data=0.0))

    def spin(self, seconds):
        self.executor.spin_once(timeout_sec=seconds)

    def close(self):
        self.executor.shutdown()
        self.node.destroy_node()
        self.context.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    io = RosInitialTargetIO()
    try:
        result = run(io, execute=args.execute)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False), flush=True)
        return (
            0
            if result["status"] in ("preflight_passed_no_commands", "initial_targets_confirmed")
            else 2
        )
    finally:
        io.close()


if __name__ == "__main__":
    raise SystemExit(main())
