"""V1 transport against installed mux in isolated ROS domain 231. No drivers."""

# ruff: noqa: E402
# Establish the isolated ROS environment before importing transport modules.
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

os.environ["ROS_DOMAIN_ID"] = "231"
os.environ["ROS_LOCALHOST_ONLY"] = "1"
for key in (
    "ROS_STATIC_PEERS",
    "ROS_AUTOMATIC_DISCOVERY_RANGE",
    "CYCLONEDDS_URI",
    "FASTRTPS_DEFAULT_PROFILES_FILE",
    "FASTDDS_DEFAULT_PROFILES_FILE",
):
    os.environ.pop(key, None)
from robot_control.closed_loop import ClosedLoop, ComputeLane, atomic_json
from robot_control.closed_loop_contracts import Cancellation, CycleIdentity, LoopConfig
from robot_control.closed_loop_native import NativeRosMotionTransport, create_session
from robot_control.closed_loop_simulation import SimulatedObservationSource, SyntheticGeometry


def plant_process(namespace, output, stop_event):
    if not namespace.startswith("/forcevla_sandbox_"):
        raise ValueError("isolated namespace required")
    import rclpy
    from marvin_msgs.msg import JointcmdArm, Jointfeedback
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from rclpy.signals import SignalHandlerOptions
    from std_msgs.msg import Float32MultiArray, Int16MultiArray, String

    context = Context()
    rclpy.init(
        context=context, domain_id=231, signal_handler_options=SignalHandlerOptions.NO
    )
    node = Node(
        "synthetic_loop_plant",
        namespace=namespace,
        context=context,
        enable_rosout=False,
        start_parameter_services=False,
    )
    executor = SingleThreadedExecutor(context=context)
    executor.add_node(node)
    spec = importlib.util.spec_from_file_location(
        "synthetic_gate",
        "/opt/kernelmind/apex_tool/install/lib/python3.12/site-packages/dm_gripper_py/web_gripper.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gate = module.Gate(output / "gripper.json")
    import numpy as np

    q = np.zeros(14)
    trace = []
    feedback = node.create_publisher(
        Jointfeedback, namespace + "/info/joint_feedback", 1
    )
    modes = node.create_publisher(Int16MultiArray, namespace + "/info/arm_state", 1)
    grip = node.create_publisher(String, namespace + "/info/web_gripper", 10)
    motors = [
        node.create_publisher(
            Float32MultiArray, namespace + "/info/gripper_feedback_" + side, 1
        )
        for side in ("L", "R")
    ]
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

    def joint_cb(side):
        def cb(msg):
            q[side * 7 : side * 7 + 7] = msg.positions
            trace.append(
                dict(time=time.monotonic(), side=side, positions=list(msg.positions))
            )

        return cb

    subs = [
        node.create_subscription(
            JointcmdArm, namespace + "/control/joint_cmd_" + side, joint_cb(i), qos
        )
        for i, side in enumerate(("A", "B"))
    ]

    def publish_gripper():
        issued = gate.issued if gate.reserved else [0.5, 0.5]
        msg = String()
        msg.data = json.dumps(
            dict(
                gate.status(),
                healthy=True,
                actual=issued,
                applied_target=issued,
                motor_feedback_age=[0.0, 0.0],
                sample_time_ns=time.time_ns(),
                calibration=[[0.0, 1.6], [0.0, 1.6]],
            )
        )
        grip.publish(msg)

    def command(msg):
        gate.command(json.loads(msg.data), [0.5, 0.5], True)
        publish_gripper()

    def heartbeat(msg):
        gate.heartbeat(json.loads(msg.data), True)
        publish_gripper()

    subs.extend(
        [
            node.create_subscription(
                String, namespace + "/control/web_gripper", command, 10
            ),
            node.create_subscription(
                String, namespace + "/control/web_gripper/heartbeat", heartbeat, 1
            ),
        ]
    )

    def tick():
        msg = Jointfeedback()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.arm_positions = q.tolist()
        msg.arm_velocities = [0.0] * 14
        feedback.publish(msg)
        mode = Int16MultiArray()
        mode.data = [1, 1]
        modes.publish(mode)
        for pub in motors:
            motor = Float32MultiArray()
            motor.data = [0.8, 0.0]
            pub.publish(motor)
        gate.tick([0.5, 0.5], True)
        publish_gripper()

    node.create_timer(0.01, tick)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    try:
        while not stop_event.wait(0.05):
            pass
    finally:
        executor.shutdown(timeout_sec=1)
        thread.join(timeout=1)
        node.destroy_node()
        context.shutdown()
        atomic_json(output / "mux-output.json", trace)


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    sys.setswitchinterval(0.0002)
    config = LoopConfig(max_task_duration_s=30)
    namespace = "/forcevla_sandbox_" + uuid.uuid4().hex[:10]
    import multiprocessing

    ctx = multiprocessing.get_context("spawn")
    plant_stop = ctx.Event()
    plant = ctx.Process(
        target=plant_process, args=(namespace, args.output, plant_stop), daemon=True
    )
    plant.start()
    mux_log = (args.output / "mux.log").open("w")
    mux = subprocess.Popen(
        [
            "/opt/kernelmind/apex/install/marvin_teleop/lib/marvin_teleop/joint_cmd_mux",
            "--ros-args",
            "-r",
            "__ns:=" + namespace,
            "-r",
            "__node:=loop_mux",
            "-p",
            "ramp_duration_sec:=3.0",
        ],
        stdout=mux_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    io = session = compute = None
    stop_supervisor = threading.Event()
    report = dict(
        source_sha256=ClosedLoop._source_hashes(),
        config=__import__("dataclasses").asdict(config),
        mode="isolated_ROS",
        synthetic=True,
        domain=231,
        namespace=namespace,
        hardware_commands_sent=0,
        hardware_execution_verified=False,
    )
    try:
        io = NativeRosMotionTransport(
            config,
            args.output / "native",
            SyntheticGeometry().bounds,
            domain=231,
            namespace=namespace,
        )
        end = time.monotonic() + 8
        while True:
            try:
                io.feedback()
                if io.selector.service_is_ready():
                    break
            except Exception:
                pass
            if time.monotonic() > end:
                raise RuntimeError("ROS discovery timeout")
            time.sleep(0.02)
        io.select_input(0)
        session = create_session(
            io, SyntheticGeometry().bounds, config, args.output / "execution"
        )

        def supervise():
            while not stop_supervisor.wait(0.01):
                session.supervise()

        threading.Thread(target=supervise, daemon=True).start()
        compute = ComputeLane("simulate", config, {})
        compute.ready(session.supervise)
        session.acquire("isolated_ROS_test")
        # Observation uses the actual subscribed mux/plant feedback, with synthetic
        # camera imagery and explicit synthetic task geometry only.
        source = SimulatedObservationSource(io, config)
        cancel = Cancellation()
        cycles = []
        for cycle in range(1, 4):
            from robot_control.closed_loop_observation import ObservationPending

            end = time.monotonic() + 2
            while True:
                try:
                    obs = source.read_after(session.barrier, None)
                    break
                except ObservationPending:
                    if time.monotonic() > end:
                        raise
                    time.sleep(0.01)
            ident = CycleIdentity("isolated_ROS", cycle, 0, obs.id, uuid.uuid4().hex)
            result = compute.call(
                "infer",
                (obs.request("synthetic transport test", 0), ident.request_id),
                2,
                session.supervise,
                cancel,
                0,
            )
            segment = compute.call(
                "prepare",
                (obs, result, ident, args.output / f"plan-{cycle}"),
                1,
                session.supervise,
                cancel,
                0,
            )
            cycles.append(session.execute(segment, cancel))
        stop = session.stop("normal_completion")
        if not stop.get("confirmed"):
            raise RuntimeError("stop unconfirmed")
        session.release()
        report.update(
            status="passed",
            cycles=cycles,
            stop=stop,
            fixture_process_separate=True,
            final_feedback=io.feedback().joints.tolist(),
        )
    except Exception as exc:
        report.update(
            status="failed",
            reason=str(exc),
            session=session.status() if session else None,
        )
        if session:
            report["stop"] = session.stop("test_failure")
    finally:
        if compute:
            compute.close()
        if session:
            session.close()
        elif io:
            io.close()
        stop_supervisor.set()
        os.killpg(mux.pid, signal.SIGINT)
        try:
            mux.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(mux.pid, signal.SIGKILL)
            mux.wait()
        mux_log.close()
        plant_stop.set()
        plant.join(timeout=3)
        if plant.is_alive():
            plant.terminate()
            plant.join(timeout=1)
        if (args.output / "mux-output.json").exists():
            report["mux_samples"] = len(
                json.loads((args.output / "mux-output.json").read_text())
            )
        trace_path = args.output / "execution/execution.jsonl"
        if trace_path.exists():
            rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
            submitted = [r for r in rows if r.get("kind") == "prediction"]
            report["prediction_submissions"] = len(submitted)
            report["max_submission_lateness_s"] = max(
                (r["time"] - r["due"] for r in submitted), default=None
            )
        native_trace = args.output / "native/native-ticks.jsonl"
        if native_trace.exists():
            ticks = [json.loads(line) for line in native_trace.read_text().splitlines()]
            predictions = [row for row in ticks if row["kind"] == 1]
            report["native_prediction_submissions"] = len(predictions)
            report["native_max_submission_lateness_s"] = max(
                (row["completed"] - row["due"] for row in predictions), default=None
            )
            report["native_timing_used"] = True
        atomic_json(args.output / "report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
