"""Re-evaluate saved predictions, no inference or ROS."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

from robot_control.common import ROOT, save_json, new_run
from robot_control.native_qp import NativeQP


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", type=Path, default=ROOT / 'data/integration-runs/20261014-shadow-final')
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = new_run(args.output, "qp-compare")
    limits = json.loads((ROOT / "config/planning.json").read_text())
    solver = NativeQP(limits["expected_qp_parameters"], limits)
    report = {
        "status": "completed_offline_comparison",
        "scope": "saved predictions, no hardware, no fresh execution permission",
        "records": [],
        "manifest": solver.manifest,
    }
    for original in sorted(args.predictions.glob("prediction_*")):
        folder = out / original.name
        folder.mkdir()
        with np.load(original / "observation.npz", allow_pickle=False) as d:
            state = d["state"].copy()
            seed = d["joint_positions_rad"].copy()
        with np.load(original / "prediction.npz", allow_pickle=False) as d:
            actions = d["actions"][0].copy()
        result = solver.rollout(actions, seed, state, folder)
        save_json(folder / "qp_checks.json", result)
        old = json.loads((original / "ik_checks.json").read_text())
        row = {
            "prediction": original.name,
            "qp_status": result["status"],
            "qp_reasons": result["reasons"],
            "previous_geometric_status": old["status"],
            "ticks": result.get("simulated_ticks"),
            "solve_wall_ms": result.get("solve_wall_ms"),
            "native_process_ms": result.get("native_process_ms"),
            "max_position_error_m": max(
                max(t["position_error_m"]) for t in result.get("targets", [])
            )
            if result.get("targets")
            else None,
            "max_rotation_error_rad": max(
                max(t["rotation_error_rad"]) for t in result.get("targets", [])
            )
            if result.get("targets")
            else None,
        }
        report["records"].append(row)
        print(json.dumps(row), flush=True)
    save_json(out / "report.json", report)


if __name__ == "__main__":
    main()
