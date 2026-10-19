"""Typed, timestamped observation assembly. No ROS or hardware writes."""

from collections import Counter, deque
from dataclasses import dataclass
import io
import threading
import time
import warnings

import cv2
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation

CAMERAS = ("image", "right_image", "wrist_image", "right_wrist_image")
FIELDS = ("left_pose", "right_pose", "left_wrench", "right_wrench", "gripper", "camera")


class Rejected(ValueError):
    pass


def finite(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != shape or not np.isfinite(a).all():
        raise Rejected("invalid_" + name)
    return a


def pose6(position, quaternion, frame):
    if frame != "base_link":
        raise Rejected("unexpected_pose_frame:" + frame)
    p = finite(position, (3,), "position")
    q = finite(quaternion, (4,), "quaternion")
    if abs(np.linalg.norm(q) - 1) > 1e-3:
        raise Rejected("nonunit_quaternion")
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        angles = Rotation.from_quat(q).as_euler("xyz", degrees=False)
    return np.concatenate((p, angles)), bool(seen)


def wrench6(force, torque, frame, expected_frame):
    if frame != expected_frame:
        raise Rejected("unexpected_wrench_frame:" + frame)
    return finite([*force, *torque], (6,), "wrench")


def gripper_values(obj, expected_calibration):
    if obj.get("healthy") is not True:
        raise Rejected("gripper_unhealthy")
    actual = finite(obj.get("actual"), (2,), "gripper_actual")
    calibration = finite(obj.get("calibration"), (2, 2), "calibration")
    if not np.array_equal(calibration, np.asarray(expected_calibration)):
        raise Rejected("gripper_calibration_changed")
    ages = finite(obj.get("motor_feedback_age"), (2,), "motor_feedback_age")
    if (ages < 0).any():
        raise Rejected("negative_motor_feedback_age")
    # Preserve slightly out-of-range actual feedback, exactly as training does.
    return actual, ages


def training_tiles(rgb):
    """Match the converter's four tiles, AREA resize, pad and JPEG95 4:4:4 round trip."""
    if rgb.dtype != np.uint8 or rgb.shape not in ((1080, 1920, 3), (2160, 3840, 3)):
        raise Rejected("unexpected_quad_shape:" + str(rgb.shape))
    height, width = rgb.shape[:2]
    out = {}
    for name, (x, y) in zip(CAMERAS, ((0, 0), (1, 0), (0, 1), (1, 1))):
        tile = rgb[
            y * height // 2 : (y + 1) * height // 2,
            x * width // 2 : (x + 1) * width // 2,
        ]
        padded = np.zeros((480, 640, 3), dtype=np.uint8)
        padded[60:420] = cv2.resize(tile, (640, 360), interpolation=cv2.INTER_AREA)
        buffer = io.BytesIO()
        Image.fromarray(padded).save(buffer, format="JPEG", quality=95, subsampling=0)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            out[name] = np.array(decoded.convert("RGB"))
    return out


def live_training_tiles(rgb, expected_shape):
    """Adapt declared live full-FOV mosaic geometry before the recorded-data transform.

    Live WebRTC and recorded H.264 encoders differ. This preserves the configured
    four views but does not assert pixel equality with a historical recording.
    """
    if list(rgb.shape) != expected_shape or rgb.dtype != np.uint8:
        raise Rejected("unexpected_live_quad_shape:" + str(rgb.shape))
    canonical = cv2.resize(rgb, (1920, 1080), interpolation=cv2.INTER_AREA)
    return training_tiles(canonical)


@dataclass(frozen=True)
class Sample:
    value: object
    source_ns: int
    receipt_ns: int
    receipt_mono_ns: int
    metadata: dict


class ObservationStore:
    """Bounded causal histories aligned to a new camera frame, independently updated."""

    def __init__(self, config):
        self.config = config
        self.fields = (
            FIELDS[:-1]
            + (("joints",) if config.get("require_joints") else ())
            + ("camera",)
        )
        self.histories = {
            k: deque(maxlen=512 if k != "camera" else 2) for k in self.fields
        }
        self.lock = threading.Lock()
        self.counts = Counter()
        self.last_camera_ns = 0
        self.errors = {}

    def reject(self, key, reason):
        with self.lock:
            self.counts["rejected:" + key + ":" + str(reason)] += 1
            self.errors[key] = str(reason)

    def update(
        self,
        key,
        value,
        source_ns,
        metadata=None,
        *,
        receipt_ns=None,
        receipt_mono_ns=None,
    ):
        wall = time.time_ns() if receipt_ns is None else receipt_ns
        mono = time.monotonic_ns() if receipt_mono_ns is None else receipt_mono_ns
        if type(source_ns) is not int or source_ns <= 0:
            raise Rejected("invalid_source_timestamp")
        limits = self.config["limits"]
        if source_ns > wall + int(limits["max_future_s"] * 1e9):
            raise Rejected("future_source_timestamp")
        if wall - source_ns > int(limits["max_source_age_s"] * 1e9):
            raise Rejected("stale_at_receipt")
        with self.lock:
            history = self.histories[key]
            if history and source_ns <= history[-1].source_ns:
                raise Rejected("duplicate_or_out_of_order_timestamp")
            # Any invalid update latches the stream unavailable until a new valid sample.
            history.append(Sample(value, source_ns, wall, mono, metadata or {}))
            self.counts["accepted:" + key] += 1
            self.errors.pop(key, None)

    def snapshot(self, *, now_ns=None, now_mono_ns=None):
        wall = time.time_ns() if now_ns is None else now_ns
        mono = time.monotonic_ns() if now_mono_ns is None else now_mono_ns
        with self.lock:
            if self.errors:
                raise Rejected("invalid_stream:" + ",".join(sorted(self.errors)))
            missing = [k for k, h in self.histories.items() if not h]
            if missing:
                raise Rejected("missing:" + ",".join(missing))
            camera = self.histories["camera"][-1]
            if camera.source_ns <= self.last_camera_ns:
                raise Rejected("no_new_camera")
            chosen = {"camera": camera}
            for key in self.fields[:-1]:
                chosen[key] = next(
                    (
                        v
                        for v in reversed(self.histories[key])
                        if v.source_ns <= camera.source_ns
                    ),
                    None,
                )
                if chosen[key] is None:
                    raise Rejected("no_causal_sample:" + key)
        limits = self.config["limits"]
        times = {}
        for key, sample in chosen.items():
            source_age = (wall - sample.source_ns) / 1e9
            receipt_age = (mono - sample.receipt_mono_ns) / 1e9
            wall_receipt_age = (wall - sample.receipt_ns) / 1e9
            if (
                source_age < -limits["max_future_s"]
                or min(receipt_age, wall_receipt_age) < 0
            ):
                raise Rejected("clock_or_future:" + key)
            if (
                source_age > limits["max_source_age_s"]
                or max(receipt_age, wall_receipt_age) > limits["max_receipt_age_s"]
            ):
                raise Rejected("stale:" + key)
            times[key] = {
                "source_ns": sample.source_ns,
                "receipt_ns": sample.receipt_ns,
                "source_age_s": source_age,
                "receipt_age_s": receipt_age,
                **sample.metadata,
            }
        skew = (
            max(s.source_ns for s in chosen.values())
            - min(s.source_ns for s in chosen.values())
        ) / 1e9
        if skew > limits["max_skew_s"]:
            raise Rejected("cross_source_skew")
        grip, motor_ages = chosen["gripper"].value
        # The reported underlying feedback ages continue to grow after its source sample.
        motor_ages_now = motor_ages + max(
            0.0, (wall - chosen["gripper"].source_ns) / 1e9
        )
        if float(motor_ages_now.max()) > limits["max_motor_feedback_age_s"]:
            raise Rejected("stale_motor_feedback")
        state = np.concatenate(
            (
                chosen["left_pose"].value,
                grip[:1],
                chosen["left_wrench"].value,
                chosen["right_pose"].value,
                grip[1:],
                chosen["right_wrench"].value,
            )
        ).astype(np.float32)
        finite(state, (26,), "state")
        request = {k: v.copy() for k, v in camera.value.items()}
        request.update(state=state, prompt=self.config["prompt"])
        with self.lock:
            self.last_camera_ns = camera.source_ns
        return request, {
            "joint_positions_rad": chosen["joints"].value.tolist()
            if "joints" in chosen
            else None,
            "camera_source_ns": camera.source_ns,
            "snapshot_unix_ns": wall,
            "snapshot_monotonic_ns": mono,
            "cross_source_skew_s": skew,
            "streams": times,
            "motor_feedback_age_s": motor_ages_now.tolist(),
            "camera_transport_equivalence_verified": self.config[
                "camera_transport_equivalence_verified"
            ],
            "camera_mapping_verified": self.config["camera_mapping_verified"],
            "physical_tcp_verified": self.config["physical_tcp_verified"],
            "wrench_physical_semantics_verified": self.config[
                "wrench_physical_semantics_verified"
            ],
        }

    def diagnostics(self):
        with self.lock:
            return {
                "counters": dict(self.counts),
                "stream_errors": dict(self.errors),
                "latest_source_ns": {
                    k: h[-1].source_ns if h else None for k, h in self.histories.items()
                },
            }
