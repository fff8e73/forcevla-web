"""Immutable observation snapshots shared by the web and CLI consumers."""

import io
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

CAMERAS = ("image", "right_image", "wrist_image", "right_wrist_image")
JOINT_NAMES = tuple(f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8))
SOURCES = ("image", "left_pose", "right_pose", "left_wrench", "right_wrench", "gripper")


@dataclass
class Snapshot:
    meta: dict
    state: np.ndarray
    images: dict

    @property
    def id(self):
        return self.meta["snapshot_id"]

    def pack(self, compressed=False):
        out = io.BytesIO()
        save = np.savez_compressed if compressed else np.savez
        save(out, metadata=json.dumps(self.meta, allow_nan=False), state=self.state, **self.images)
        return out.getvalue()

    @classmethod
    def unpack(cls, data):
        with np.load(io.BytesIO(data), allow_pickle=False) as raw:
            return cls(
                json.loads(str(raw["metadata"])),
                raw["state"].copy(),
                {k: raw[k].copy() for k in CAMERAS if k in raw},
            )

    def jpeg(self, camera):
        out = io.BytesIO()
        Image.fromarray(self.images[camera]).save(out, format="JPEG", quality=88)
        return out.getvalue()

    def validate(self, config, now=None, wall=None):
        now = time.monotonic() if now is None else now
        wall = time.time() if wall is None else wall
        reasons = list(self.meta.get("errors", []))
        sources = self.meta.get("sources", {})
        receipts, stamps = [], []
        ages = {}
        for name in SOURCES:
            src = sources.get(name)
            if not src:
                reasons.append(f"{name}: 等待数据")
                continue
            age = now - src["received_mono"]
            if not np.isfinite(age):
                reasons.append(f"{name}: 接收时间不是有限值")
                ages[name] = None
                continue
            ages[name] = age
            receipts.append(src["received_mono"])
            if age < 0 or age > config["max_age_s"]:
                reasons.append(f"{name}: 数据过期 ({age:.2f}s)")
            stamp = src.get("source_time")
            if stamp is None or not np.isfinite(stamp) or stamp <= 0:
                reasons.append(f"{name}: 源时间不可验证")
            else:
                stamps.append(stamp)
                if not -0.1 <= wall - stamp <= config["max_source_age_s"]:
                    reasons.append(f"{name}: 源采样时间过期或时钟不一致")
        skew = max(receipts) - min(receipts) if receipts else None
        source_skew = max(stamps) - min(stamps) if stamps else None
        if skew is not None and skew > config["max_skew_s"]:
            reasons.append(f"接收时间偏差过大 ({skew:.3f}s)")
        if source_skew is not None and source_skew > config["max_skew_s"]:
            reasons.append(f"源时间偏差过大 ({source_skew:.3f}s)")
        motor = self.meta.get("motor_feedback_age", [])
        grip_age = ages.get("gripper")
        if grip_age is None:
            grip_age = float("inf")
        if len(motor) != 2 or any(
            v is None or not np.isfinite(v) or v < 0 or v + grip_age > config["max_age_s"]
            for v in motor
        ):
            reasons.append("夹爪底层反馈缺失或过期")
        if (
            self.state.shape != (26,)
            or self.state.dtype != np.float32
            or not np.isfinite(self.state).all()
        ):
            reasons.append("26维状态缺失或非有限值")
        for key in CAMERAS:
            image = self.images.get(key)
            if image is None or image.shape != (480, 640, 3) or image.dtype != np.uint8:
                reasons.append(f"{key}: 图像缺失或格式错误")
        return {
            "ready": not reasons,
            "reasons": reasons,
            "ages": ages,
            "skew_s": skew,
            "source_skew_s": source_skew,
            "exposure_sync": "未验证（拼图仅提供统一时间戳）",
            "transport": self.meta.get("transport", {}),
        }


class FileSource:
    def __init__(self, path):
        self.path = Path(path)
        self.last = None
        self.signature = None

    def read(self):
        try:
            stat = self.path.stat()
            signature = (stat.st_ino, stat.st_mtime_ns)
            if signature != self.signature:
                candidate = Snapshot.unpack(self.path.read_bytes())
                self.last, self.signature = candidate, signature
        except FileNotFoundError:
            return None
        return self.last


class DemoSource:
    """Explicit synthetic inputs, never a fallback for failed live data."""

    def read(self):
        tick = int(time.monotonic() * 5)
        now, wall = time.monotonic(), time.time()
        images = {}
        for i, key in enumerate(CAMERAS):
            arr = np.zeros((480, 640, 3), dtype=np.uint8)
            arr[60:420, :, i % 3] = np.arange(640, dtype=np.uint16)[None, :] % 180 + 35
            arr[100:160, (tick * 8) % 540 : (tick * 8) % 540 + 100, :] = 220
            images[key] = arr
        state = np.zeros(26, dtype=np.float32)
        state[:3] = [0.35, 0.22, 0.8]
        state[13:16] = [0.35, -0.22, 0.8]
        state[[6, 19]] = 0.4
        return Snapshot(
            {
                "snapshot_id": f"demo-{tick}",
                "created_mono": now,
                "created_unix": wall,
                "sources": {k: {"received_mono": now, "source_time": wall} for k in SOURCES},
                "motor_feedback_age": [0.01, 0.01],
                "errors": [],
                "synthetic": True,
            },
            state,
            images,
        )


def align_feedback(histories, image_source):
    """Select complete feedback samples nearest the image source clock."""
    values, sources, motor_ages = {}, {}, []
    if image_source is None:
        return values, sources, motor_ages
    sources["image"] = dict(image_source)
    for key, history in histories.items():
        if history:
            value, source, motor = min(
                history, key=lambda row: abs(row[1]["source_time"] - image_source["source_time"])
            )
            values[key], sources[key] = value, source
            if key == "gripper":
                motor_ages = motor
    return values, sources, motor_ages


def ordered_joints(names, positions):
    """Reorder by explicit joint name; never assume publisher array order."""
    if len(names) != len(positions) or len(set(names)) != len(names):
        raise ValueError("Joint names/positions mismatch or duplicate names")
    lookup = dict(zip(names, positions))
    missing = [name for name in JOINT_NAMES if name not in lookup]
    if missing:
        raise ValueError("Missing joints: " + ", ".join(missing))
    result = np.asarray([lookup[name] for name in JOINT_NAMES], dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite joint feedback")
    return result
