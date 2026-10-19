"""Explain a saved Stage-C rejection offline; never refresh freshness or execute."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from robot_control.action_adapter import prepare_actions
from robot_control.common import ROOT, new_run, save_json
from robot_control.native_qp import NativeQP
from robot_control.prepared_joint_plan import PreparedJointPlan
from robot_control.motion_prefix import check_motion_prefix

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--source", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
args = p.parse_args()
limits = json.loads((ROOT / "config/planning.json").read_text())
profile = json.loads((ROOT / "config/stage_d_trial.json").read_text())
out = new_run(args.output, "saved-prefix-diagnosis")
solver = NativeQP(limits["expected_qp_parameters"], limits)
with np.load(args.source / "observation.npz", allow_pickle=False) as d:
    seed = d["joint_positions_rad"].copy()
    state = d["state"].copy()
    velocity = d["seed_velocity_rad_s"].copy()
with np.load(args.source / "prediction.npz", allow_pickle=False) as d:
    raw = d["actions"].copy()
metadata = json.loads((args.source / "observation_metadata.json").read_text())
candidate, adaptation = prepare_actions(raw, limits["max_gripper_correction"])
if candidate is None:
    raise ValueError("No bounded candidate")
check = solver.rollout(candidate[0], seed, state, out)
save_json(out / "qp_checks.json", check)
report = {
    "scope": "offline analysis of saved data, never executable or fresh",
    "hardware_commands_sent": 0,
    "qp_status": check["status"],
    "qp_reasons": check["reasons"],
    "original_observed_at": metadata["original_observed_at"],
}
if check["status"] == "passed_scoped_qp":
    with np.load(out / "qp_trajectory.npz") as d:
        keep = d["target_indices"] < profile["prefix_steps"]
        q = d["joint_positions"][keep].copy()
        ids = d["target_indices"][keep]
        dt = float(d["dt"])
    plan = PreparedJointPlan(
        metadata["request_id"],
        metadata["original_observed_at"],
        seed,
        q,
        candidate[0, ids][:, [6, 13]],
        dt,
        {"prefix_steps": profile["prefix_steps"]},
    )
    report["prefix_check"] = check_motion_prefix(
        plan,
        solver,
        profile,
        seed_velocity=velocity,
        initial_gripper_targets=np.asarray(
            metadata["streams"]["gripper"]["applied_target"]
        ),
    )
save_json(out / "report.json", report)
print(json.dumps(report, ensure_ascii=False, indent=2))
