"""Diagnostic gates for frozen predictions; never produces controller commands."""

import numpy as np
from robot_control.offline_ik import pose_delta


def check_actions(actions, state, age_s, limits):
    a = np.asarray(actions, dtype=np.float64)
    s = np.asarray(state, dtype=np.float64)
    if a.shape != (1, 50, 14) or not np.isfinite(a).all():
        raise ValueError("Invalid physical actions")
    if s.shape != (26,) or not np.isfinite(s).all():
        raise ValueError("Invalid observation state")
    if not np.isfinite(age_s) or age_s < 0:
        raise ValueError("Invalid observation age")
    a = a[0]
    records = []
    reasons = []
    if age_s > limits["max_response_age_s"]:
        reasons.append("response_stale")
    for i, row in enumerate(a):
        current = [pose_delta(row[:6], s[:6]), pose_delta(row[7:13], s[13:19])]
        previous = [s[:6], s[13:19]] if i == 0 else [a[i - 1, :6], a[i - 1, 7:13]]
        delta = [pose_delta(row[:6], previous[0]), pose_delta(row[7:13], previous[1])]
        why = []
        if any(p > limits["max_displacement_m"] for p, r in current):
            why.append("tcp_displacement")
        if any(r > limits["max_rotation_rad"] for p, r in current):
            why.append("tcp_rotation")
        if any(p > limits["max_tcp_step_m"] for p, r in delta):
            why.append("tcp_step_translation")
        if any(r > limits["max_tcp_step_rad"] for p, r in delta):
            why.append("tcp_step_rotation")
        if (row[[6, 13]] < 0).any() or (row[[6, 13]] > 1).any():
            why.append("gripper_out_of_range")
        records.append(
            {
                "target_index": i,
                "from_observation_translation_m": [p for p, r in current],
                "from_observation_rotation_rad": [r for p, r in current],
                "step_translation_m": [p for p, r in delta],
                "step_rotation_rad": [r for p, r in delta],
                "gripper": row[[6, 13]].tolist(),
                "reasons": why,
            }
        )
        reasons.extend(why)
    return {
        "status": "passed_diagnostic_gates" if not reasons else "rejected",
        "reasons": sorted(set(reasons)),
        "response_observation_age_s": age_s,
        "nominal_last_target_age_s": age_s + 49 / 30,
        "targets": records,
        "thresholds_scope": "provisional prediction diagnostic budgets; not approved hardware safety thresholds",
    }


def check_analysis_age(age_s, maximum_s):
    if (
        not np.isfinite(age_s)
        or age_s < 0
        or not np.isfinite(maximum_s)
        or maximum_s <= 0
    ):
        raise ValueError("Invalid post-analysis age")
    return [] if age_s <= maximum_s else ["analysis_result_stale"]
