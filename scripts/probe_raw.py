"""Read-only four-camera timing/shape probe; no ROS control interfaces."""

import json
import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

root = Path("artifacts/stability-20261014")
root.mkdir(parents=True, exist_ok=True)
rclpy.init()
node = rclpy.create_node("forcevla_four_raw_probe")
cv2.setNumThreads(2)
names = ["left_eye", "right_eye", "left_wrist", "right_wrist"]
counts = {k: 0 for k in names}
last, gaps, age, conversions = {}, {k: [] for k in names}, {k: [] for k in names}, []
last_convert, stamps, skews = {}, {}, []
shapes = {}


def callback(name):
    def receive(m):
        now = time.monotonic()
        counts[name] += 1
        stamp = m.header.stamp.sec + m.header.stamp.nanosec / 1e9
        if name in last:
            gaps[name].append(now - last[name])
        last[name] = now
        age[name].append(time.time() - stamp)
        stamps[name] = stamp
        shapes[name] = [m.width, m.height, m.step, len(m.data), m.encoding, m.header.frame_id]
        if now - last_convert.get(name, 0) < 0.18:
            return
        last_convert[name] = now
        start = time.monotonic()
        frame = np.frombuffer(m.data, dtype=np.uint8).reshape(m.height * 3 // 2, m.step)[
            :, : m.width
        ]
        rgb = cv2.cvtColor(frame, cv2.COLOR_YUV2RGB_NV12)
        resized = cv2.resize(rgb, (640, 360), interpolation=cv2.INTER_AREA)
        padded = np.pad(resized, ((60, 60), (0, 0), (0, 0)))
        conversions.append(time.monotonic() - start)
        if not (root / f"raw-{name}.png").exists():
            cv2.imwrite(str(root / f"raw-{name}.png"), cv2.cvtColor(padded, cv2.COLOR_RGB2BGR))
        if len(stamps) == 4:
            skews.append(max(stamps.values()) - min(stamps.values()))

    return receive


subs = [
    node.create_subscription(
        Image, f"/camera/{k}/image_nv12_undistorted", callback(k), qos_profile_sensor_data
    )
    for k in names
]
started = time.monotonic()
try:
    while time.monotonic() - started < 20:
        rclpy.spin_once(node, timeout_sec=0.02)
finally:
    node.destroy_node()
    rclpy.shutdown()


def metrics(values):
    return (
        {
            "count": len(values),
            "p50": float(np.median(values)),
            "p95": float(np.percentile(values, 95)),
            "max": float(max(values)),
        }
        if values
        else {}
    )


report = {
    "duration_s": time.monotonic() - started,
    "counts": counts,
    "shapes": shapes,
    "receive_gap_s": {k: metrics(v) for k, v in gaps.items()},
    "source_age_s": {k: metrics(v) for k, v in age.items()},
    "cross_camera_skew_s": metrics(skews),
    "rgb_resize_s": metrics(conversions),
}
(root / "raw-probe.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
