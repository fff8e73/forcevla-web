"""Post-hold observations with explicit wall/monotonic clock mapping."""

import copy
import json
import time
from pathlib import Path

import numpy as np

from robot_control.closed_loop_contracts import Observation
from robot_control.command_bridge import ControlFault

CAMERAS = ("image", "right_image", "wrist_image", "right_wrist_image")
SOURCES = ("image", "left_pose", "right_pose", "left_wrench", "right_wrench", "gripper")
JOINT_NAMES = [f"Joint{i}_{side}" for side in ("L", "R") for i in range(1, 8)]


class ObservationPending(ControlFault):
    pass


def validated_observation(
    meta, state, images, barrier, previous_id, config, *, now=None, wall=None
):
    now = time.monotonic() if now is None else now
    wall = time.time() if wall is None else wall
    if not np.isfinite([now, wall, barrier.monotonic, barrier.unix]).all():
        raise ControlFault("invalid_clock")
    if (
        now < barrier.monotonic
        or abs((wall - now) - (barrier.unix - barrier.monotonic))
        > config.clock_tolerance_s
    ):
        raise ControlFault("observation_clock_changed")
    if meta.get("errors") or not meta.get("snapshot_id"):
        raise ObservationPending("incomplete_observation")
    if meta["snapshot_id"] == previous_id:
        raise ObservationPending("observation_id_repeated")
    sources = {k: meta.get("sources", {}).get(k, {}) for k in SOURCES}
    joint = meta.get("joints", {})
    cf = meta.get("control_feedback", {})
    velocity = cf.get("joints", {})
    if joint.get("error") or velocity.get("error") or joint.get("names") != JOINT_NAMES:
        raise ObservationPending("invalid_joint_feedback")
    sources["joints"] = joint.get("source", {})
    sources["gripper_target"] = meta.get("gripper_target_source", {})
    sources["joint_velocity"] = velocity
    source_times, receipt_times = {}, []
    for key, src in sources.items():
        stamp, received = src.get("source_time"), src.get("received_mono")
        if (
            not isinstance(stamp, (int, float))
            or not isinstance(received, (int, float))
            or not np.isfinite([stamp, received]).all()
        ):
            raise ObservationPending("missing_source_clock:" + key)
        if stamp <= barrier.unix or received <= barrier.monotonic:
            raise ObservationPending("sample_before_hold:" + key)
        if (
            not 0 <= now - received <= config.max_observation_age_s
            or not 0 <= wall - stamp <= config.max_observation_age_s
        ):
            raise ObservationPending("sample_stale_or_future:" + key)
        source_times[key] = stamp
        receipt_times.append(received)
    if (
        max(source_times.values()) - min(source_times.values())
        > config.max_source_skew_s
    ):
        raise ObservationPending("source_skew")
    if max(receipt_times) - min(receipt_times) > config.max_source_skew_s:
        raise ObservationPending("receipt_skew")
    # Driver ages bind the unstamped motor topic to a verified timestamped status.
    # A newly received copy of an old motor report cannot cross the barrier.
    ages = np.asarray(meta.get("motor_feedback_age"), dtype=float)
    if ages.shape != (2,) or not np.isfinite(ages).all() or np.any(ages < 0):
        raise ObservationPending("invalid_motor_ages")
    motor_sources = source_times["gripper"] - ages
    if np.any(motor_sources <= barrier.unix) or np.any(
        wall - motor_sources > config.max_observation_age_s
    ):
        raise ObservationPending("motor_before_hold_or_stale")
    for index, key in enumerate(("left_gripper", "right_gripper")):
        motor = cf.get(key, {})
        received = motor.get("received_mono", float("nan"))
        if (
            motor.get("error")
            or not np.isfinite(received)
            or not barrier.monotonic < received <= now
            or now - received > config.feedback_age_s
        ):
            raise ObservationPending("motor_feedback_missing_or_old:" + key)
        if not np.isfinite(
            [motor.get("position_rad", np.nan), motor.get("velocity_rad_s", np.nan)]
        ).all():
            raise ObservationPending("motor_nonfinite")
        source_times[key] = float(motor_sources[index])
    q = np.array(joint.get("positions"), dtype=float, copy=True)
    v = np.array(velocity.get("velocities"), dtype=float, copy=True)
    velocity_q = np.asarray(velocity.get("positions"), dtype=float)
    state = np.array(state, dtype=np.float32, copy=True)
    if (
        q.shape != (14,)
        or v.shape != (14,)
        or velocity_q.shape != (14,)
        or state.shape != (26,)
        or not all(np.isfinite(x).all() for x in (q, v, velocity_q, state))
    ):
        raise ObservationPending("invalid_state_arrays")
    if np.max(np.abs(q - velocity_q)) > config.seed_error_rad:
        raise ObservationPending("joint_position_velocity_misaligned")
    targets = np.asarray(meta.get("gripper_applied_target"), dtype=float)
    if (
        targets.shape != (2,)
        or not np.isfinite(targets).all()
        or np.any(targets < 0)
        or np.any(targets > 1)
    ):
        raise ObservationPending("missing_applied_gripper_target")
    frozen_images = {}
    for key in CAMERAS:
        image = np.array(images.get(key), copy=True)
        if image.shape != (480, 640, 3) or image.dtype != np.uint8:
            raise ObservationPending("invalid_image:" + key)
        image.flags.writeable = False
        frozen_images[key] = image
    for a in (state, q, v, targets):
        a.flags.writeable = False
    # Conservative mapping: earliest represented source sample, including motors.
    observed_at = now - (wall - min(source_times.values()))
    return Observation(
        meta["snapshot_id"],
        observed_at,
        source_times,
        state,
        q,
        v,
        targets,
        frozen_images,
        copy.deepcopy(meta),
    )


class FileObservationSource:
    def __init__(self, path, config):
        self.path, self.config = Path(path), config

    def read_after(self, barrier, previous_id):
        try:
            with np.load(self.path, allow_pickle=False) as data:
                meta = json.loads(str(data["metadata"]))
                return validated_observation(
                    meta,
                    data["state"],
                    {key: data[key] for key in CAMERAS},
                    barrier,
                    previous_id,
                    self.config,
                )
        except FileNotFoundError as exc:
            raise ObservationPending("collector_unavailable") from exc
