"""Capture a short camera-only recording and compare both live image branches.

Requires --record-sample. No robot command APIs are used. Existing recording and
nonzero robot input mode prevent this diagnostic from starting a new recording.
"""

import argparse
import json
import time
import urllib.request
from collections import deque
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
import rclpy
import rosbag2_py
from marvin_msgs.srv import VideoCapture
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import CompressedImage


def ssim(a, b):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float64)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float64)
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    va = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a * mu_a
    vb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b * mu_b
    cov = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    return float(
        np.mean(
            ((2 * mu_a * mu_b + 6.5025) * (2 * cov + 58.5225))
            / ((mu_a * mu_a + mu_b * mu_b + 6.5025) * (va + vb + 58.5225))
        )
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-sample", action="store_true", required=True)
    parser.add_argument("--output", default="artifacts/stability-20261014/image-contract")
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    session_dir = root / f"camera-only-{int(time.time())}"
    for path, field, expected in [
        ("/api/v1/record/status", "is_recording", False),
        ("/api/v1/robot/status", "current_input_mode", 0),
    ]:
        data = json.load(urllib.request.urlopen("http://127.0.0.1:8080" + path, timeout=5))["data"]
        if data[field] != expected:
            raise RuntimeError("Camera-only capture requires idle recording and robot input None")
    report = {
        "criteria": {
            "pairs_required": 5,
            "max_source_skew_s": 0.045,
            "per_view_ssim_min": 0.90,
            "per_view_rgb_mae_max": 8.0,
        },
        "pairs": [],
        "camera_only_recording": str(session_dir),
        "robot_motion_commands": False,
    }
    rclpy.init()
    node = rclpy.create_node("forcevla_training_image_comparison")
    cv2.setNumThreads(2)
    frames = {k: deque(maxlen=30) for k in ["preview", "record"]}
    codecs = {k: av.CodecContext.create("h264", "r") for k in frames}
    last = {k: 0.0 for k in frames}
    counts = {k: 0 for k in frames}
    for decoder in codecs.values():
        decoder.thread_count = 2

    def receive(key):
        def cb(msg):
            stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
            packet = av.Packet(bytes(msg.data))
            packet.pts = round(stamp * 1e9)
            packet.time_base = Fraction(1, 10**9)
            try:
                decoded = codecs[key].decode(packet)
            except av.FFmpegError:
                return
            counts[key] += len(decoded)
            if decoded and stamp - last[key] > 0.5:
                frame = decoded[-1]
                if frame.is_corrupt:
                    return
                last[key] = stamp
                frames[key].append((frame.pts / 1e9, frame.to_ndarray(format="rgb24")))

        return cb

    qos = QoSProfile(depth=32, reliability=ReliabilityPolicy.BEST_EFFORT)
    subs = [
        node.create_subscription(CompressedImage, topic, receive(key), qos)
        for key, topic in [
            ("preview", "/quad_tile/compressed_undistorted"),
            ("record", "/recorder/quad_tile/compressed_undistorted"),
        ]
    ]
    client = node.create_client(VideoCapture, "/recorder/set_recording")
    owned = False
    try:
        if not client.wait_for_service(timeout_sec=5):
            raise RuntimeError("Camera recorder service unavailable")
        request = VideoCapture.Request()
        request.start_stop = True
        request.save_dir = str(session_dir)
        future = client.call_async(request)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        if not future.done() or not future.result().success:
            raise RuntimeError("Camera recorder did not confirm start")
        owned = True
        report["start_response"] = future.result().message
        deadline = time.monotonic() + 12
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.01)
        report["decoded_frames"] = counts
        report["geometry_photometric_check_passed"] = len(report["pairs"]) >= 5 and all(
            v["ssim"] >= 0.90 and v["rgb_mae"] <= 8 for p in report["pairs"] for v in p["views"]
        )
        report["pixel_identical"] = False
        report["physical_camera_mapping_verified"] = False
    finally:
        if owned:
            request = VideoCapture.Request()
            request.start_stop = False
            request.save_dir = str(session_dir)
            future = client.call_async(request)
            rclpy.spin_until_future_complete(node, future, timeout_sec=15)
            report["stop_confirmed"] = bool(future.done() and future.result().success)
            if future.done():
                report["stop_response"] = future.result().message
        for sub in subs:
            node.destroy_subscription(sub)
        node.destroy_node()
        rclpy.shutdown()
        (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    if not report.get("stop_confirmed"):
        raise RuntimeError("Camera recording stop was not confirmed")
    bags = list(session_dir.rglob("metadata.yaml"))
    if len(bags) != 1:
        raise RuntimeError("Expected exactly one camera-only recording")
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bags[0].parent), storage_id="mcap"),
        rosbag2_py.ConverterOptions("", ""),
    )
    decoder = av.CodecContext.create("h264", "r")
    decoder.thread_count = 2
    used = set()
    while reader.has_next() and len(report["pairs"]) < 5:
        topic, data, bag_ns = reader.read_next()
        if topic != "/recorder/quad_tile/compressed_undistorted":
            continue
        msg = deserialize_message(data, CompressedImage)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec / 1e9
        packet = av.Packet(bytes(msg.data))
        packet.pts = round(stamp * 1e9)
        packet.time_base = Fraction(1, 10**9)
        try:
            decoded = decoder.decode(packet)
        except av.FFmpegError:
            continue
        counts["record"] += len(decoded)
        if not decoded or not frames["preview"]:
            continue
        frame = decoded[-1]
        if frame.is_corrupt:
            continue
        tb = frame.pts / 1e9
        ta, a = min(frames["preview"], key=lambda row: abs(row[0] - tb))
        skew = abs(ta - tb)
        if ta in used or skew > 0.045:
            continue
        b = frame.to_ndarray(format="rgb24")
        if a.shape != (1488, 1920, 3) or b.shape != (1080, 1920, 3):
            raise ValueError(f"Unexpected image dimensions: {a.shape}, {b.shape}")
        used.add(ta)
        resized = cv2.resize(a, (1920, 1080), interpolation=cv2.INTER_AREA)
        index = len(report["pairs"])
        views = []
        for y, x in [(0, 0), (0, 960), (540, 0), (540, 960)]:
            aa = cv2.resize(
                resized[y : y + 500, x : x + 960], (480, 250), interpolation=cv2.INTER_AREA
            )
            bb = cv2.resize(b[y : y + 500, x : x + 960], (480, 250), interpolation=cv2.INTER_AREA)
            views.append(
                {
                    "ssim": ssim(aa, bb),
                    "rgb_mae": float(np.abs(aa.astype(float) - bb.astype(float)).mean()),
                }
            )
        report["pairs"].append(
            {
                "preview_stamp": ta,
                "record_stamp": tb,
                "source_skew_s": skew,
                "views": views,
                "full_image_rgb_mae": float(np.abs(resized.astype(float) - b.astype(float)).mean()),
            }
        )
        if index == 0:
            cv2.imwrite(str(root / "preview-resized.png"), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(root / "record-native.png"), cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
            np.savez_compressed(
                root / "paired-rgb.npz",
                preview=resized,
                record=b,
                preview_stamp=ta,
                record_stamp=tb,
            )
    report["decoded_frames"] = counts
    report["geometry_photometric_check_passed"] = len(report["pairs"]) >= 5 and all(
        v["ssim"] >= 0.90 and v["rgb_mae"] <= 8 for p in report["pairs"] for v in p["views"]
    )
    report["comparison_source"] = (
        "timestamp-paired preview ROS stream vs closed camera MCAP recording"
    )
    (root / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
