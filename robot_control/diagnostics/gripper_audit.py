"""Read-only gripper lineage audit: dataset, frozen references, saved RPC arrays.

Run with a local Python containing NumPy/PyArrow. No ROS, inference or control.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def summary(values):
    a = np.concatenate(values, axis=0).reshape(-1, 2)
    return {
        "rows": len(a),
        "min": a.min(axis=0).tolist(),
        "max": a.max(axis=0).tolist(),
        "below_zero": (a < 0).sum(axis=0).tolist(),
        "above_one": (a > 1).sum(axis=0).tolist(),
        "q01": np.quantile(a, 0.01, axis=0).tolist(),
        "q99": np.quantile(a, 0.99, axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runtime", type=Path, default=Path("/home/user/forcevla-runtime-thor")
    )
    p.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/media/user/BAG_STORAGE/forcevla_real_data/ethernet_insert_inputForce_success"
        ),
    )
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path(
            "/home/user/forcevla-web/data/integration-runs/20261014-shadow-final"
        ),
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    stats_path = (
        args.runtime
        / "converted-job4252-v1/assets/ethernet_insert_inputForce_success/norm_stats.json"
    )
    stats = json.loads(stats_path.read_text())["norm_stats"]["actions"]
    mean = np.array(stats["mean"])[[6, 13]]
    std = np.array(stats["std"])[[6, 13]]
    out = {
        "started_unix_ns": time.time_ns(),
        "scope": "read_only_gripper_lineage_not_hardware_acceptance",
        "statistics_sha256": sha(stats_path),
        "action_gripper_mean": mean.tolist(),
        "action_gripper_std": std.tolist(),
    }
    shutil.copy2(__file__, args.output / "gripper_audit.py")
    out["audit_script_sha256"] = sha(__file__)
    reference_report = args.runtime / "baseline-job5444/report.json"
    reference = json.loads(reference_report.read_text())
    if reference.get("status") != "passed" or reference.get("samples") != 30:
        raise ValueError("Invalid frozen reference manifest")
    frozen_hashes = {r["file"]: r["sha256"] for r in reference["records"]}
    out["frozen_manifest_sha256"] = sha(reference_report)
    references = []
    native = []
    truth = []
    feedback = []
    for file in sorted((args.runtime / "baseline-job5444").glob("sample_*.npz")):
        if sha(file) != frozen_hashes[file.name]:
            raise ValueError("Frozen sample hash changed: " + file.name)
        with np.load(file, allow_pickle=False) as d:
            references.append(d["physical_actions"][:, [6, 13]])
            truth.append(d["ground_truth_actions"][:, [6, 13]])
            feedback.append(d["raw_state"][None, [6, 19]])
        with np.load(
            args.runtime / "artifacts/service-025/direct" / file.name,
            allow_pickle=False,
        ) as d:
            native.append(d["actions"][0][:, [6, 13]])
    out["frozen_original"] = summary(references)
    out["accepted_native"] = summary(native)
    out["frozen_ground_truth"] = summary(truth)
    out["frozen_actual_feedback"] = summary(feedback)
    reports = []
    for folder in sorted(args.predictions.glob("prediction_*")):
        with np.load(folder / "prediction.npz", allow_pickle=False) as d:
            actual = d["actions"][0][:, [6, 13]].copy()
            z = d["normalized_actions"][0][:, [6, 13]].copy()
        with np.load(folder / "observation.npz", allow_pickle=False) as d:
            state = d["state"][[6, 19]].copy()
        expected = (z.astype(np.float64) * (std + 1e-6) + mean).astype(np.float32)
        reports.append(
            {
                "prediction": folder.name,
                "prediction_sha256": sha(folder / "prediction.npz"),
                "actual_feedback": state.tolist(),
                "output": summary([actual]),
                "recomputed_denormalization_max_abs": float(
                    np.max(np.abs(actual - expected))
                ),
                "output_motor_radian_min": (actual.min(axis=0) * 1.6).tolist(),
                "normalized_zero_threshold": (-mean / (std + 1e-6)).tolist(),
            }
        )
    out["saved_live_predictions"] = reports
    actions = []
    states = []
    dataset_files = []
    for file in sorted((args.dataset / "data").rglob("*.parquet")):
        table = pq.read_table(file, columns=["action", "observation.state"])
        a = np.asarray(table["action"].to_pylist(), dtype=float)
        s = np.asarray(table["observation.state"].to_pylist(), dtype=float)
        if (
            a.ndim != 2
            or a.shape[1] != 14
            or s.shape != (len(a), 26)
            or not np.isfinite(a).all()
            or not np.isfinite(s).all()
        ):
            raise ValueError("Invalid dataset numeric contract")
        actions.append(a[:, [6, 13]])
        states.append(s[:, [6, 19]])
        dataset_files.append(
            {
                "file": str(file),
                "rows": len(a),
                "numeric_sha256": hashlib.sha256(a.tobytes() + s.tobytes()).hexdigest(),
            }
        )
    if not dataset_files:
        raise ValueError("No training dataset parquet files")
    out["dataset"] = {
        "path": str(args.dataset),
        "episodes": len(dataset_files),
        "action": summary(actions),
        "actual_feedback": summary(states),
        "files": dataset_files,
        "provenance_limit": "Current local dataset, not proof of byte-identical remote training-job data. Frozen original predictions are checkpoint-specific evidence.",
    }
    out["findings"] = {
        "denormalization_exact_for_saved_predictions": all(
            r["recomputed_denormalization_max_abs"] == 0 for r in reports
        ),
        "original_checkpoint_outputs_also_out_of_range": any(
            out["frozen_original"][k][j] > 0
            for k in ["below_zero", "above_one"]
            for j in range(2)
        ),
        "dataset_actions_out_of_range": any(
            out["dataset"]["action"][k][j] > 0
            for k in ["below_zero", "above_one"]
            for j in range(2)
        ),
        "no_actions_changed": True,
        "no_controller_calls": True,
    }
    out["finished_unix_ns"] = time.time_ns()
    (args.output / "report.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    print(
        json.dumps(
            {
                k: out[k]
                for k in [
                    "findings",
                    "frozen_original",
                    "accepted_native",
                    "frozen_ground_truth",
                ]
            },
            indent=2,
        )
    )
    print(
        "dataset",
        json.dumps(
            {k: v for k, v in out["dataset"].items() if k != "files"},
            ensure_ascii=False,
        ),
    )


if __name__ == "__main__":
    main()
