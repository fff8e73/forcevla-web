"""ROS-only subprocess. Subscription APIs only; no actuator SDK or publishers."""

import argparse
import io
import json
import os
import threading
import time
import uuid
from collections import OrderedDict, deque
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from PIL import Image
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32MultiArray, String

from forcevla_web.observation import CAMERAS, JOINT_NAMES, Snapshot, align_feedback, ordered_joints
from forcevla_web.transport import camera_transport


def split_tile(rgb, expected_shape=(1080, 1920, 3)):
    if rgb is None or rgb.shape != tuple(expected_shape):
        raise ValueError(
            f"Unexpected tile shape: {None if rgb is None else rgb.shape}; expected {expected_shape}"
        )
    if rgb.shape[:2] != (1080, 1920):
        rgb = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_AREA)
    images = {}
    for key, (y, x) in zip(CAMERAS, ((0, 0), (0, 960), (540, 0), (540, 960))):
        view = cv2.resize(rgb[y : y + 540, x : x + 960], (640, 360), interpolation=cv2.INTER_AREA)
        padded = np.pad(view, ((60, 60), (0, 0), (0, 0)))
        encoded = io.BytesIO()
        Image.fromarray(padded).save(encoded, format="JPEG", quality=95, subsampling=0)
        encoded.seek(0)
        images[key] = np.asarray(Image.open(encoded).convert("RGB")).copy()
    return images


class TileDecoder:
    def __init__(self, expected_shape, output_period_s=0.18):
        if (
            type(output_period_s) not in (int, float)
            or not np.isfinite(output_period_s)
            or not 0 < output_period_s <= 1
        ):
            raise ValueError("Invalid decoded image output period")
        self.output_period_s = float(output_period_s)
        self.expected_shape = expected_shape
        self.codec = av.CodecContext.create("h264", "r")
        self.codec.thread_count = 2
        self.receipts = OrderedDict()
        self.last_output = 0.0
        self.stats = {
            "packets": 0,
            "frames": 0,
            "outputs": 0,
            "no_frame": 0,
            "throttled": 0,
            "stamp_gaps": 0,
        }
        self.last_stamp = None

    def decode_frame(self, msg, received):
        """Decode every compressed access unit, without RGB/JPEG work on this lane."""
        stamp_ns = msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec
        self.stats["packets"] += 1
        if self.last_stamp and stamp_ns - self.last_stamp > 60000000:
            self.stats["stamp_gaps"] += 1
        self.last_stamp = stamp_ns
        if "h264" in msg.format.lower():
            packet = av.Packet(bytes(msg.data))
            packet.pts = packet.dts = stamp_ns
            packet.time_base = Fraction(1, 10**9)
            self.receipts[stamp_ns] = received
            while len(self.receipts) > 64:
                self.receipts.popitem(last=False)
            frames = self.codec.decode(packet)
            self.stats["frames"] += len(frames)
            if not frames:
                self.stats["no_frame"] += 1
                return None
            frame = frames[-1]
            if frame.is_corrupt:
                raise ValueError("Corrupt H.264 frame; waiting for recovery")
            if frame.pts is None or frame.pts not in self.receipts:
                raise ValueError("Decoded frame has no traceable source timestamp")
            return frame, self.receipts[frame.pts], frame.pts / 1e9
        bgr = cv2.imdecode(np.frombuffer(msg.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("Invalid JPEG/PNG image")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), received, stamp_ns / 1e9

    def render_frame(self, decoded):
        frame, received, stamp = decoded
        rgb = frame.to_ndarray(format="rgb24") if isinstance(frame, av.VideoFrame) else frame
        images = split_tile(rgb, self.expected_shape)
        self.stats["outputs"] += 1
        return images, received, stamp

    def decode(self, msg, received):
        """Synchronous helper for tests and offline clients, preserving timestamp tracing."""
        decoded = self.decode_frame(msg, received)
        if decoded is None:
            return None
        if received - self.last_output < self.output_period_s:
            self.stats["throttled"] += 1
            return None
        self.last_output = received
        return self.render_frame(decoded)


class LatestDecodedFrame:
    """One already-decoded frame; replacing it cannot corrupt predictive decoding."""

    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.replaced = 0

    def put(self, decoded):
        with self.condition:
            if self.pending is not None:
                self.replaced += 1
            self.pending = decoded
            self.condition.notify()

    def take(self, timeout=.1):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None, timeout=timeout)
            value, self.pending = self.pending, None
            return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    config = json.loads(Path(parser.parse_args().config).read_text())
    path = Path(config["ipc_path"])
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    rclpy.init()
    node = rclpy.create_node("forcevla_web_observer")
    camera_node = rclpy.create_node("forcevla_web_camera")
    image_lock = threading.RLock()
    qos = QoSProfile(
        depth=1, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE
    )
    values, sources, images, errors = {}, {}, {}, {}
    control_feedback = {}
    motor_ages = []
    counters = {}
    transport = {}
    histories = {
        key: deque(maxlen=100)
        for key in ("left_pose", "right_pose", "left_wrench", "right_wrench", "gripper", "joints")
    }
    boot, seq = uuid.uuid4().hex[:12], 0
    decoder = TileDecoder(config["image_shape"], config.get("image_output_period_s", 0.18))
    latest_decoded = LatestDecodedFrame()
    processing_stop = threading.Event()
    processing_stats = {}
    comparison_path = config.get("image_comparison_report")
    comparison = {}
    if comparison_path and Path(comparison_path).is_file():
        comparison = json.loads(Path(comparison_path).read_text())
    compared = comparison.get("geometry_photometric_check_passed") is True
    image_warning = (
        "2026-10-14预览/录制画面几何与颜色对照通过；编码非逐像素相同，现场映射与曝光同步待验"
        if compared
        else "当前使用1920×1488预览流整图缩放；与训练录制流的视野/编码等价性待验证"
    )
    cv2.setNumThreads(1)

    def callback(key):
        def receive(msg):
            nonlocal images, motor_ages
            now = time.monotonic()
            counters[key] = counters.get(key, 0) + 1
            if key in ("left_pose", "right_pose", "left_wrench", "right_wrench", "joints"):
                previous = sources.get(key)
                if previous and now - previous["received_mono"] < 0.02:
                    return
            try:
                if key == "gripper":
                    raw = json.loads(msg.data)
                    values[key] = np.asarray(raw["actual"], dtype=np.float32)
                    if values[key].shape != (2,) or not np.isfinite(values[key]).all():
                        raise ValueError("Invalid gripper actual")
                    motor_ages = raw.get("motor_feedback_age", [])
                    if not raw.get("healthy", False):
                        raise ValueError("Gripper reports unhealthy")
                    stamp = raw["sample_time_ns"] / 1e9
                    applied = np.asarray(raw.get("applied_target"), dtype=float)
                    if applied.shape != (2,) or not np.isfinite(applied).all():
                        raise ValueError("Invalid applied gripper target")
                    control_feedback["gripper_target"] = {
                        "positions": applied.tolist(),
                        "source_time": stamp,
                        "received_mono": now,
                    }
                else:
                    stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
                    if key == "image":
                        begun = time.monotonic()
                        decoded = decoder.decode_frame(msg, now)
                        processing_stats["decode_ms"] = (time.monotonic() - begun) * 1000
                        processing_stats["packet_source_age_s"] = time.time() - stamp
                        if decoded is not None:
                            latest_decoded.put(decoded)
                        return
                    elif key == "joints":
                        values[key] = ordered_joints(list(msg.name), list(msg.position))
                        try:
                            velocity = ordered_joints(list(msg.name), list(msg.velocity))
                            control_feedback["joints"] = {
                                "positions": values[key].tolist(),
                                "velocities": velocity.tolist(),
                                "received_mono": now,
                                "source_time": stamp,
                            }
                        except Exception as exc:
                            control_feedback["joints"] = {"error": str(exc), "received_mono": now}
                    elif "pose" in key:
                        if msg.header.frame_id != "base_link":
                            raise ValueError("Unexpected pose frame")
                        p, q = msg.pose.position, msg.pose.orientation
                        values[key] = np.r_[
                            [p.x, p.y, p.z],
                            Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz"),
                        ]
                    else:
                        expected = "flange_L" if key.startswith("left") else "flange_R"
                        if msg.header.frame_id != expected:
                            raise ValueError("Unexpected wrench frame")
                        f, t = msg.wrench.force, msg.wrench.torque
                        values[key] = np.array([f.x, f.y, f.z, t.x, t.y, t.z])
                counters[key + "_accepted"] = counters.get(key + "_accepted", 0) + 1
                sources[key] = {"received_mono": now, "source_time": stamp}
                histories[key].append((values[key].copy(), dict(sources[key]), list(motor_ages)))
                errors.pop(key, None)
            except Exception as exc:
                with image_lock:
                    errors[key] = f"{key}: {exc}"

        return receive

    specs = [
        ("image", CompressedImage, config["image_topic"]),
        ("left_pose", PoseStamped, "/tj/info/eef_left"),
        ("right_pose", PoseStamped, "/tj/info/eef_right"),
        ("left_wrench", WrenchStamped, "/tj/info/wrench_left"),
        ("right_wrench", WrenchStamped, "/tj/info/wrench_right"),
        ("gripper", String, "/info/web_gripper"),
        ("joints", JointState, "/tj/joint_states"),
    ]
    subscriptions = [
        (camera_node if key == "image" else node).create_subscription(
            kind,
            topic,
            callback(key),
            QoSProfile(depth=16, reliability=ReliabilityPolicy.BEST_EFFORT)
            if key == "image"
            else qos,
        )
        for key, kind, topic in specs
    ]

    def control_gripper_callback(side):
        def receive(msg):
            now = time.monotonic()
            try:
                raw = np.asarray(msg.data, dtype=float)
                if raw.size < 2 or not np.isfinite(raw[:2]).all():
                    raise ValueError("Invalid motor position/velocity")
                control_feedback[side] = {
                    "position_rad": float(raw[0]),
                    "velocity_rad_s": float(raw[1]),
                    "received_mono": now,
                }
            except Exception as exc:
                control_feedback[side] = {"error": str(exc), "received_mono": now}

        return receive

    for side, topic in [
        ("left_gripper", "/info/gripper_feedback_L"),
        ("right_gripper", "/info/gripper_feedback_R"),
    ]:
        subscriptions.append(
            node.create_subscription(Float32MultiArray, topic, control_gripper_callback(side), qos)
        )

    def flush_locked():
        nonlocal seq
        seq += 1
        selected_values, selected_sources, selected_motor = align_feedback(
            histories, sources.get("image")
        )
        state = np.full(26, np.nan, dtype=np.float32)
        for key, sl in (
            ("left_pose", slice(0, 6)),
            ("left_wrench", slice(7, 13)),
            ("right_pose", slice(13, 19)),
            ("right_wrench", slice(20, 26)),
        ):
            if key in selected_values:
                state[sl] = selected_values[key]
        if "gripper" in selected_values:
            state[[6, 19]] = selected_values["gripper"]
        joint_source = selected_sources.pop("joints", None)
        joint_meta = (
            {
                "names": list(JOINT_NAMES),
                "positions": selected_values["joints"].tolist(),
                "source": joint_source,
                "error": errors.get("joints"),
            }
            if "joints" in selected_values
            else {"error": errors.get("joints", "等待实际关节反馈")}
        )
        meta = {
            "control_feedback": dict(control_feedback),
            "gripper_applied_target": control_feedback.get("gripper_target", {}).get("positions"),
            "gripper_target_source": {
                k: control_feedback.get("gripper_target", {}).get(k)
                for k in ("source_time", "received_mono")
            },
            "joints": joint_meta,
            "snapshot_id": f"{boot}-{seq}",
            "created_mono": time.monotonic(),
            "created_unix": time.time(),
            "sources": selected_sources,
            "alignment": "nearest source timestamp from bounded feedback history",
            "motor_feedback_age": selected_motor,
            "errors": [value for key, value in errors.items() if key != "joints"],
            "source_counts": dict(counters),
            "decoder_stats": dict(decoder.stats),
            "image_processing": dict(processing_stats, decoded_frames_replaced=latest_decoded.replaced,
                                     policy="decode_every_packet_render_latest_frame"),
            "transport": dict(transport),
            "image_topic": config["image_topic"],
            "source_image_shape": config["image_shape"],
            "image_contract_verified": False,
            "image_warning": image_warning,
            "image_comparison": {
                "historical_geometry_photometric_passed": compared,
                "report_path": comparison_path,
                "pixel_identical": False,
            },
            "image_contract": "undistorted tile / 1920x1080 / quadrants / area640x360 / pad60 / RGB",
            "jpeg_roundtrip": "quality95/subsampling0",
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_bytes(Snapshot(meta, state, images).pack())
        os.replace(tmp, path)

    def flush():
        with image_lock:
            flush_locked()

    def inspect_transport():
        nonlocal transport
        try:
            report = camera_transport(camera_node, config["image_topic"])
        except Exception as exc:
            report = {"warning": f"相机连接诊断暂不可用: {exc}"}
        with image_lock:
            transport = report

    diagnostic_timer = node.create_timer(5.0, inspect_transport)
    snapshot_period = config.get("observation_period_s", 0.2)
    if (
        type(snapshot_period) not in (int, float)
        or not np.isfinite(snapshot_period)
        or not 0.02 <= snapshot_period <= 1
    ):
        raise ValueError("Invalid observation period")
    timer = node.create_timer(snapshot_period, flush)
    camera_executor = SingleThreadedExecutor()
    camera_executor.add_node(camera_node)
    observer_executor = SingleThreadedExecutor()
    observer_executor.add_node(node)

    def spin_camera():
        try:
            camera_executor.spin()
        except ExternalShutdownException:
            pass

    def process_latest_image():
        nonlocal images
        due = 0.0
        while not processing_stop.is_set():
            if processing_stop.wait(max(0, due - time.monotonic())):
                return
            decoded = latest_decoded.take()
            if decoded is None:
                continue
            started = time.monotonic()
            try:
                next_images, received, stamp = decoder.render_frame(decoded)
                finished = time.monotonic()
                with image_lock:
                    images = next_images
                    sources["image"] = {"received_mono": received, "source_time": stamp}
                    counters["image_accepted"] = counters.get("image_accepted", 0) + 1
                    processing_stats["render_ms"] = (finished-started)*1000
                    processing_stats["rendered_source_age_s"] = time.time()-stamp
                    errors.pop("image", None)
            except Exception as exc:
                with image_lock:
                    errors["image"] = f"image: {exc}"
            due = started + decoder.output_period_s

    processing_thread = threading.Thread(target=process_latest_image, name="image-converter", daemon=True)
    processing_thread.start()
    image_thread = threading.Thread(target=spin_camera, name="camera-decoder", daemon=True)
    image_thread.start()
    try:
        observer_executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_timer(timer)
        node.destroy_timer(diagnostic_timer)
        camera_executor.shutdown(timeout_sec=3)
        observer_executor.shutdown(timeout_sec=3)
        image_thread.join(timeout=3)
        processing_stop.set()
        processing_thread.join(timeout=3)
        for (key, _, _), sub in zip(specs, subscriptions):
            (camera_node if key == "image" else node).destroy_subscription(sub)
        camera_node.destroy_node()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
