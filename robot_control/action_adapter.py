"""Explicit execution-boundary gripper saturation; preserves raw model arrays.

Only action columns 6 and 13 may change. TCP values and observed feedback are
never clamped. Large corrections and nonfinite values remain rejected.
"""

import math
import numpy as np


def prepare_actions(raw_actions, max_gripper_correction):
    if (
        type(max_gripper_correction) not in (int, float)
        or not math.isfinite(max_gripper_correction)
        or not 0 <= max_gripper_correction <= 1
    ):
        raise ValueError("Invalid gripper correction budget")
    raw = np.asarray(raw_actions)
    if (
        raw.shape != (1, 50, 14)
        or not np.issubdtype(raw.dtype, np.floating)
        or not np.isfinite(raw).all()
    ):
        raise ValueError("Invalid finite floating 1x50x14 physical prediction")
    bounded = raw.copy()
    bounded[..., [6, 13]] = np.clip(raw[..., [6, 13]], 0.0, 1.0)
    original = raw[0][:, [6, 13]]
    command = bounded[0][:, [6, 13]]
    delta = command.astype(np.float64) - original.astype(np.float64)
    maximum = float(np.abs(delta).max())
    accepted = maximum <= max_gripper_correction
    report = {
        "policy": "gripper_saturation_v1",
        "status": "prepared" if accepted else "rejected",
        "reasons": [] if accepted else ["gripper_correction_excessive"],
        "raw_shape": list(raw.shape),
        "gripper_columns": [6, 13],
        "max_allowed_correction": max_gripper_correction,
        "max_correction": maximum,
        "changed_values": int(np.count_nonzero(delta)),
        "changed_values_per_arm": np.count_nonzero(delta, axis=0).tolist(),
        "raw_min": original.min(axis=0).astype(float).tolist(),
        "raw_max": original.max(axis=0).astype(float).tolist(),
        "bounded_min": command.min(axis=0).astype(float).tolist(),
        "bounded_max": command.max(axis=0).astype(float).tolist(),
        "raw_gripper": original.astype(float).tolist(),
        "bounded_gripper": command.astype(float).tolist(),
        "corrections": delta.tolist(),
        "tcp_unchanged": True,
        "sensor_feedback_modified": False,
        "scope": "execution candidate only, not hardware approval or a change to model numerical acceptance",
    }
    return bounded if accepted else None, report
