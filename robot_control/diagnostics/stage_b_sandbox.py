"""Stage B against installed mux + installed pure gripper Gate in domain231.

Starts no drivers, SDK or hardware clients. Cannot choose production endpoints.
The plant is idealized. Routing/hold acknowledgements do not prove motor stop.
"""

import argparse
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import numpy as np

from robot_control.common import new_run, save_json, sha256
from robot_control.joint_execution import JointExecutionBackend
from robot_control.joint_source_guard import JointEnvelope
from robot_control.ros_execution_transport import RosExecutionTransport

MUX = "/opt/kernelmind/apex/install/marvin_teleop/lib/marvin_teleop/joint_cmd_mux"
GATE = "/opt/kernelmind/apex_tool/install/lib/python3.12/site-packages/dm_gripper_py/web_gripper.py"


def main(stage_d=False, measured=False):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--fault", choices=(["none", "right_write", "upstream_loss", "manual_abort", "deadline", "stale_feedback"]
                 if stage_d else ["none", "right_write", "upstream_loss"]), default="none"
    )
    args = parser.parse_args()
    output = new_run(args.output, "stage-d" if stage_d else "stage-b")
    namespace = "/forcevla_sandbox_" + uuid.uuid4().hex[:12]
    for key in (
        "ROS_STATIC_PEERS",
        "ROS_AUTOMATIC_DISCOVERY_RANGE",
        "CYCLONEDDS_URI",
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
    ):
        os.environ.pop(key, None)
    os.environ.update(ROS_DOMAIN_ID="231", ROS_LOCALHOST_ONLY="1")
    report = {
        "status": "failed",
        "domain": 231,
        "namespace": namespace,
        "mux_sha256": sha256(MUX),
        "gate_sha256": sha256(GATE),
        "hardware_commands_sent": 0,
        "moving_stop_verified": False,
        "fault": args.fault,
    }
    process = node = executor = ctx = transport = backend = thread = None
    log = None
    try:
        import rclpy
        from marvin_msgs.msg import JointcmdArm, Jointfeedback
        from rclpy.context import Context
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy
        from rclpy.signals import SignalHandlerOptions
        from std_msgs.msg import Float32MultiArray, Int16MultiArray, String

        spec = importlib.util.spec_from_file_location("sandbox_gripper_gate", GATE)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        gate = module.Gate(output / "synthetic_gripper_lease.json")
        ctx = Context()
        rclpy.init(
            args=[],
            context=ctx,
            domain_id=231,
            signal_handler_options=SignalHandlerOptions.NO,
        )
        node = Node(
            "synthetic_plant",
            namespace=namespace,
            context=ctx,
            enable_rosout=False,
            start_parameter_services=False,
        )
        executor = SingleThreadedExecutor(context=ctx)
        executor.add_node(node)
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
        feedback = node.create_publisher(
            Jointfeedback, namespace + "/info/joint_feedback", 1
        )
        mode = node.create_publisher(Int16MultiArray, namespace + "/info/arm_state", 1)
        grip = node.create_publisher(String, namespace + "/info/web_gripper", 10)
        motors = [
            node.create_publisher(
                Float32MultiArray, namespace + "/info/gripper_feedback_" + s, 1
            )
            for s in ("L", "R")
        ]
        seed_pubs = [
            node.create_publisher(
                JointcmdArm, namespace + "/control/joint_cmd_" + s, qos
            )
            for s in ("A", "B")
        ]
        q = np.full(14, 0.1)
        outputs = [0, 0]
        output_trace = []
        previous_q = q.copy()
        previous_sample = [time.monotonic()]
        seed_mode = [not measured]

        def joint_cb(index):
            def cb(m):
                q[index * 7 : index * 7 + 7] = m.positions
                outputs[index] += 1
                if stage_d:
                    output_trace.append(dict(time=time.monotonic(),side=index,positions=list(m.positions)))

            return cb

        subs = [
            node.create_subscription(
                JointcmdArm, namespace + "/control/joint_cmd_" + s, joint_cb(i), qos
            )
            for i, s in enumerate(("A", "B"))
        ]

        def publish_grip():
            issued = gate.issued if gate.reserved else [0.5, 0.5]
            data = dict(
                gate.status(),
                healthy=True,
                actual=issued,
                applied_target=issued,
                motor_feedback_age=[0.0, 0.0],
                sample_time_ns=time.time_ns(),
                calibration=[[0.0, 1.6], [0.0, 1.6]],
            )
            msg = String()
            msg.data = json.dumps(data)
            grip.publish(msg)

        def command(m):
            gate.command(json.loads(m.data), [0.5, 0.5], True)
            publish_grip()

        def heartbeat(m):
            gate.heartbeat(json.loads(m.data), True)
            publish_grip()

        subs += [
            node.create_subscription(
                String, namespace + "/control/web_gripper", command, 10
            ),
            node.create_subscription(
                String, namespace + "/control/web_gripper/heartbeat", heartbeat, 1
            ),
        ]

        def tick():
            msg = Jointfeedback()
            msg.header.stamp = node.get_clock().now().to_msg()
            msg.arm_positions = q.tolist()
            if stage_d:
                now=time.monotonic()
                msg.arm_velocities=((q-previous_q)/max(now-previous_sample[0],1e-6)).tolist()
                previous_q[:]=q
                previous_sample[0]=now
            else:
                msg.arm_velocities = [0.0] * 14
            feedback.publish(msg)
            m = Int16MultiArray()
            m.data = [1, 1]
            mode.publish(m)
            for pub in motors:
                m = Float32MultiArray()
                m.data = [0.8, 0.0]
                pub.publish(m)
            gate.tick([0.5, 0.5], True)
            publish_grip()
            if seed_mode[0]:
                for i, pub in enumerate(seed_pubs):
                    m = JointcmdArm()
                    m.header.stamp = node.get_clock().now().to_msg()
                    m.positions = q[i * 7 : i * 7 + 7].tolist()
                    pub.publish(m)

        node.create_timer(0.01, tick)
        thread = threading.Thread(target=executor.spin, daemon=True)
        thread.start()
        log = (output / "mux.log").open("w")
        process = subprocess.Popen(
            [
                MUX,
                "--ros-args",
                "-r",
                "__ns:=" + namespace,
                "-r",
                "__node:=stage_b_mux",
                "-p",
                "ramp_duration_sec:=3.0",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        transport = RosExecutionTransport(
            domain=231, namespace=namespace, timeout_s=0.8, allow_missing_targets=measured
        )
        deadline = time.monotonic() + 8
        while True:
            try:
                transport.feedback()
                if transport.selector.service_is_ready():
                    break
            except Exception as exc:  # noqa: BLE001 - preserve transport failures in report
                report["last_discovery_issue"] = str(exc)
            if time.monotonic() > deadline:
                raise RuntimeError("sandbox_discovery_timeout")
            time.sleep(0.02)
        transport.select_input(0)
        backend_class = JointExecutionBackend
        if stage_d:
            from robot_control.segment_executor import BufferedSandboxBackend
            backend_class = BufferedSandboxBackend
        if measured:
            from robot_control.measured_hold import MeasuredHoldBackend
            backend_class = MeasuredHoldBackend
        backend = backend_class(
            transport,
            np.tile([-2.0, 2.0], (14, 1)),
            np.ones(14),
            output / "execution.json",
            lease_s=0.5,
            settle_s=0.1,
            prediction_age_s=10. if measured else 2.5,
        )
        if measured:
            from measured_hold_probe import run_cycle
            backend.start_watchdog()
            report.update(run_cycle(backend,transport,output))
            save_json(output / "report.json",report)
            return 0
        token = backend.acquire("stage_b_sandbox")
        seed_mode[0] = False
        if stage_d:
            from stage_d_sandbox import run_motion_scenario
            report.update(run_motion_scenario(backend, transport, token, args.fault, output, output_trace))
        else:
            observed = time.monotonic()
            for i in range(8):
                time.sleep(0.02)
                backend.heartbeat(token)
                now = time.monotonic()
                backend.write(JointEnvelope(token, i, now, observed, np.full(14, 0.1)))
            # Exercise separate gripper target and receipt (no real motors).
            time.sleep(0.02)
            backend.heartbeat(token)
            backend.write(
                JointEnvelope(token, 8, time.monotonic(), observed, np.full(14, 0.1)),
                [0.501, 0.501],
            )
            before = list(outputs)
            if args.fault == "right_write":
                original_publish = transport.publish_arm

                def failing_publish(side, positions):
                    if side == "right":
                        raise ConnectionError("injected_right_transport_failure")
                    original_publish(side, positions)

                transport.publish_arm = failing_publish
                time.sleep(0.02)
                backend.heartbeat(token)
                try:
                    backend.write(
                        JointEnvelope(
                            token, 9, time.monotonic(), observed, np.full(14, 0.1)
                        )
                    )
                    raise RuntimeError("fault_did_not_reject")
                except ConnectionError:
                    pass
                if (
                    backend.state != "FAULT_UNCONFIRMED"
                    or gate.state != "held"
                    or not gate.reserved
                ):
                    raise RuntimeError("partial_failure_did_not_latch_and_hold")
            elif args.fault == "upstream_loss":
                backend.start_watchdog()
                deadline = time.monotonic() + 1
                while backend.state == "ACTIVE" and time.monotonic() < deadline:
                    time.sleep(0.02)
                if backend.state == "ACTIVE":
                    raise RuntimeError("watchdog_failed_to_trip")
            else:
                backend.trip("manual_hold")
            deadline = time.monotonic() + 2
            while backend.state == "STOPPING" and time.monotonic() < deadline:
                time.sleep(0.02)
                backend.poll()
            if args.fault != "right_write":
                if backend.state != "HOLD_CONFIRMED":
                    raise RuntimeError(
                        "hold_not_confirmed:" + json.dumps(backend.stop_report)
                    )
                backend.release(token)
            report.update(
                status="passed",
                final_state=backend.state,
                outputs=outputs,
                before_hold_outputs=before,
                stop_report=backend.stop_report,
                input_after_release=transport.feedback().selected_input,
                gripper_reserved_after_release=transport.gripper_status()["reserved"],
                transport_publications=transport.publications,
                scope="installed mux + installed pure Gate + ideal synthetic plant; stationary path",
            )
            if not all(n > 0 for n in outputs) or (
                args.fault != "right_write" and report["input_after_release"] != 0
            ):
                raise RuntimeError("output_or_release_check_failed")
    except Exception as exc:  # noqa: BLE001 - preserve transport failures in report
        report["status"] = "failed"
        report["error"] = repr(exc)
    finally:
        if backend:
            backend.close()
        elif transport:
            transport.close()
        if process and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=2)
        if executor:
            executor.shutdown(timeout_sec=1)
        if thread:
            thread.join(timeout=1)
        if node:
            node.destroy_node()
        if ctx and ctx.ok():
            ctx.shutdown()
        if log:
            log.close()
        save_json(output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
