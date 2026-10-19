"""Exercise the installed mux binary in domain 231 + random namespace only.

No robot driver/SDK is started. All input and output is synthetic. Domain 24 and
namespace /tj are unconditionally disallowed; there is no live-control option.
"""

import argparse
import json
import shutil
import numpy as np
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from robot_control.common import new_run, save_json, sha256
from robot_control.joint_source_guard import JointSourceGuard, JointEnvelope, GateRejected

DOMAIN = 231
EXECUTABLE = (
    "/opt/kernelmind/apex/install/marvin_teleop/lib/marvin_teleop/joint_cmd_mux"
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ramp-duration", type=float, default=3.0)
    args = p.parse_args()
    if not 0.01 <= args.ramp_duration <= 5:
        p.error("ramp duration must be in [.01,5]")
    out = new_run(args.output, "mux-sandbox")
    shutil.copy2(__file__, out / "sandbox_mux_probe.py")
    namespace = "/forcevla_sandbox_" + uuid.uuid4().hex[:12]
    assert DOMAIN != 24 and namespace.startswith("/forcevla_sandbox_")
    env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), ROS_LOCALHOST_ONLY="1")
    # Remove externally supplied discovery routes for this isolated test.
    env.pop("ROS_STATIC_PEERS", None)
    env.pop("ROS_AUTOMATIC_DISCOVERY_RANGE", None)
    env.pop("CYCLONEDDS_URI", None)
    env.pop("FASTRTPS_DEFAULT_PROFILES_FILE", None)
    os.environ.update(ROS_DOMAIN_ID=str(DOMAIN), ROS_LOCALHOST_ONLY="1")
    for key in [
        "ROS_STATIC_PEERS",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
    ]:
        os.environ.pop(key, None)
    report = {
        "status": "failed",
        "scope": "installed_mux_binary_synthetic_only",
        "ros_domain_id": DOMAIN,
        "namespace": namespace,
        "ramp_duration_s": args.ramp_duration,
        "binary_sha256": sha256(EXECUTABLE),
        "hardware_processes_started": 0,
        "live_tj_commands_sent": 0,
        "stages": {},
    }
    process = node = context = executor = None
    log = None
    try:
        import rclpy
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.signals import SignalHandlerOptions
        from marvin_msgs.msg import JointcmdArm, Jointfeedback
        from marvin_msgs.srv import Int
        from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy

        context = Context()
        rclpy.init(
            args=[],
            context=context,
            domain_id=DOMAIN,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        node = Node(
            "probe",
            namespace=namespace,
            context=context,
            enable_rosout=False,
            start_parameter_services=False,
        )
        executor = SingleThreadedExecutor(context=context)
        executor.add_node(node)
        qos = QoSProfile(
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        outputs = {"A": [], "B": []}

        def callback(side):
            def receive(message):
                outputs[side].append(
                    {"time": time.monotonic(), "positions": list(message.positions)}
                )

            return receive

        for side in ["A", "B"]:
            node.create_subscription(
                JointcmdArm,
                namespace + "/control/joint_cmd_" + side,
                callback(side),
                QoSProfile(
                    depth=100,
                    reliability=ReliabilityPolicy.BEST_EFFORT,
                    durability=DurabilityPolicy.VOLATILE,
                ),
            )
        pubs = {}
        for source in ["user", "qp_controller"]:
            for side in ["A", "B"]:
                pubs[(source, side)] = node.create_publisher(
                    JointcmdArm,
                    namespace + "/control/" + source + "/joint_cmd_" + side,
                    qos,
                )
        feedback_pub = node.create_publisher(
            Jointfeedback, namespace + "/info/joint_feedback", qos
        )
        client = node.create_client(Int, namespace + "/control/set_input")
        log = (out / "mux.log").open("w")
        process = subprocess.Popen(
            [
                EXECUTABLE,
                "--ros-args",
                "-r",
                "__ns:=" + namespace,
                "-r",
                "__node:=isolated_mux",
                "-p",
                "ramp_duration_sec:=" + str(args.ramp_duration),
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        started = time.monotonic()
        while not client.service_is_ready():
            if time.monotonic() - started > 8:
                raise RuntimeError("Sandbox service discovery timeout")
            if process.poll() is not None:
                raise RuntimeError("Sandbox mux exited")
            executor.spin_once(timeout_sec=0.05)

        def stamp(message, old=False):
            message.header.stamp = node.get_clock().now().to_msg()
            if old:
                message.header.stamp.sec = max(1, message.header.stamp.sec - 60)

        gate = None
        token = None
        gate_observed = None
        gate_sequence = 0
        gate_rejections = {}

        def pump(duration, source_values=None, old=False):
            nonlocal gate_sequence
            end = time.monotonic() + duration
            next_send = 0.0
            while time.monotonic() < end:
                if time.monotonic() >= next_send:
                    feedback = Jointfeedback()
                    stamp(feedback)
                    feedback.arm_positions = [0.1] * 14
                    feedback_pub.publish(feedback)
                    for source, values in (source_values or {}).items():
                        if gate is not None and source == "user":
                            now = time.monotonic()
                            try:
                                gate.heartbeat(token, now, 3)
                                envelope = JointEnvelope(
                                    token,
                                    gate_sequence,
                                    now - 60 if old else now,
                                    gate_observed,
                                    np.array([values[0]] * 7 + [values[1]] * 7),
                                )
                                gate.accept(envelope, now, 3, feedback_time=now)
                                gate_sequence += 1
                            except GateRejected as exc:
                                gate_rejections[str(exc)] = (
                                    gate_rejections.get(str(exc), 0) + 1
                                )
                                continue
                        for side, value in zip(["A", "B"], values):
                            m = JointcmdArm()
                            stamp(m, old)
                            m.positions = [float(value)] * 7
                            pubs[(source, side)].publish(m)
                    next_send = time.monotonic() + 0.02
                executor.spin_once(timeout_sec=0.005)

        def select(index):
            request = Int.Request()
            request.data = index
            future = client.call_async(request)
            deadline = time.monotonic() + 2
            while not future.done() and time.monotonic() < deadline:
                executor.spin_once(timeout_sec=0.02)
            if not future.done():
                raise RuntimeError("Sandbox set_input timeout")
            result = future.result()
            return {
                "success": result.success,
                "error_code": result.error_code,
                "message": result.message,
            }

        def stage(name, mode, duration, values=None, old=False):
            if mode in (1, 3) and values or name == "user_refresh":
                duration = max(duration, args.ramp_duration + 0.35)
            response = select(mode) if mode is not None else None
            before = {s: len(outputs[s]) for s in outputs}
            begin = time.monotonic()
            pump(duration, values, old)
            result = {
                "requested_input": mode,
                "service_response": response,
                "duration_s": duration,
                "source_values": values,
                "old_header": old,
                "arms": {},
            }
            for side in outputs:
                samples = outputs[side][before[side] :]
                tail = [
                    v for v in samples if v["time"] - begin > min(0.25, duration / 2)
                ]
                result["arms"][side] = {
                    "count": len(samples),
                    "tail_count": len(tail),
                    "last_positions": samples[-1]["positions"] if samples else None,
                    "tail_min_first_joint": min(v["positions"][0] for v in tail)
                    if tail
                    else None,
                    "tail_max_first_joint": max(v["positions"][0] for v in tail)
                    if tail
                    else None,
                }
            report["stages"][name] = result
            print(name, json.dumps(result), flush=True)

        pump(0.4)
        stage(
            "user_selected", 3, 0.6, {"user": [0.2, 0.3], "qp_controller": [0.8, 0.9]}
        )
        stage("input_silence", None, 0.8)
        stage("old_user_header", None, 0.6, {"user": [0.25, 0.35]}, True)
        stage(
            "none_selected", 0, 0.5, {"user": [0.4, 0.5], "qp_controller": [0.8, 0.9]}
        )
        stage(
            "teleop_selected", 1, 0.6, {"user": [0.4, 0.5], "qp_controller": [0.6, 0.7]}
        )
        stage("switch_back_user_without_new_user_message", 3, 0.5)
        stage("user_refresh", None, 0.5, {"user": [0.22, 0.32]})
        gate = JointSourceGuard(
            np.tile([-2.0, 2.0], (14, 1)), np.ones(14), lease_s=0.25
        )
        gate_observed = time.monotonic()
        token = gate.claim(
            "sandbox_forcevla",
            gate_observed,
            np.array([0.22] * 7 + [0.32] * 7),
            gate_observed,
            3,
        )
        stage("guarded_user_valid", None, 0.5, {"user": [0.22, 0.32]})
        stage("guarded_old_header", None, 0.5, {"user": [0.25, 0.35]}, True)
        report["guard"] = {
            "state": gate.state,
            "reason": gate.reason,
            "rejections": gate_rejections,
            "hardware_stop_confirmed": False,
        }
        stages = report["stages"]
        for name, expected in [
            ("user_selected", [0.2, 0.3]),
            ("teleop_selected", [0.6, 0.7]),
            ("user_refresh", [0.22, 0.32]),
            ("guarded_user_valid", [0.22, 0.32]),
        ]:
            for side, value in zip(["A", "B"], expected):
                actual = stages[name]["arms"][side]["last_positions"]
                if actual is None or max(abs(x - value) for x in actual) > 1e-5:
                    raise RuntimeError("Unexpected routing: " + name + side)
        report["observations"] = {
            "user_input_3_routes_user": True,
            "teleop_input_1_routes_qp": True,
            "none_stops_output_in_this_test": all(
                a["tail_count"] == 0 for a in stages["none_selected"]["arms"].values()
            ),
            "silent_source_still_publishes": any(
                a["tail_count"] > 0 for a in stages["input_silence"]["arms"].values()
            ),
            "old_header_accepted": all(
                abs(stages["old_user_header"]["arms"][s]["last_positions"][0] - v)
                < 1e-5
                for s, v in [("A", 0.25), ("B", 0.35)]
            ),
            "hardware_stop_semantics_verified": False,
            "exclusive_owner_lease_verified": False,
        }
        if gate.state != "TRIPPED" or gate.reason != "command_stale_or_future":
            raise RuntimeError("Guard did not latch stale command")
        if any(v["tail_count"] for v in stages["guarded_old_header"]["arms"].values()):
            raise RuntimeError("Guard forwarded stale command")
        report["observations"]["guard_rejects_stale_command"] = True
        report["status"] = "passed_sandbox_routing_probe"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if process:
            os.killpg(process.pid, signal.SIGTERM) if process.poll() is None else None
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
            report["sandbox_process_returncode"] = process.returncode
        if executor:
            executor.shutdown(timeout_sec=1)
        if node:
            node.destroy_node()
        if context and context.ok():
            context.shutdown()
        if log:
            log.close()
        save_json(out / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(out / "report.json"),
                "error": report.get("error"),
            }
        )
    )
    return 0 if report["status"] == "passed_sandbox_routing_probe" else 1


if __name__ == "__main__":
    raise SystemExit(main())
