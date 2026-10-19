"""Stage C: prepare one finite joint prefix from fresh live observations, no writes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import uuid

import numpy as np
from PIL import Image

from robot_control.common import ROOT, load_config, new_run, save_json, sha256, source_manifest
from robot_control.observation import ObservationStore, Rejected, CAMERAS
from robot_control.ros_observer import ROSObserver
from robot_control.native_qp import NativeQP
from robot_control.prepared_joint_plan import prepare_native_plan
from robot_control.model_rpc import predict_observation
from robot_control.action_adapter import prepare_actions
from robot_control.prediction_checks import check_actions
from robot_control.motion_prefix import (
    check_motion_prefix,
    validate_trial_profile,
    hardware_trial_blockers,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    p.add_argument("--planning-config", type=Path, default=ROOT / "config/planning.json")
    p.add_argument(
        "--trial-profile", type=Path, default=ROOT / "config/stage_d_trial.json"
    )
    p.add_argument("--preflight-report", type=Path, required=True)
    p.add_argument("--stage-b-report", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--diagnostic-prefix-steps", type=int, choices=range(1,51))
    p.add_argument("--duration", type=float, default=30)
    args = p.parse_args()
    if args.seed < 0 or not 0 < args.duration <= 120:
        p.error("Invalid seed or observation wait")
    config = load_config(args.config)
    limits = json.loads(args.planning_config.read_text())
    trial = json.loads(args.trial_profile.read_text())
    validate_trial_profile(trial)
    hardware_trial = dict(trial)
    if args.diagnostic_prefix_steps is not None:
        trial["prefix_steps"] = args.diagnostic_prefix_steps
        trial["scope"] = "Explicit diagnostic-only prefix; cannot enable hardware execution"
    diagnostic_only = args.diagnostic_prefix_steps is not None
    validate_trial_profile(trial, diagnostic_only=diagnostic_only)
    out = new_run(args.output, "prepare-C")
    report = {
        "status": "failed",
        "started_unix_ns": time.time_ns(),
        "source_sha256": source_manifest(),
        "config_sha256": sha256(args.config),
        "planning_config_sha256": sha256(args.planning_config),
        "trial_profile_sha256": sha256(args.trial_profile),
        "operator_confirmation": config.get("operator_confirmation", {}),
        "hardware_commands_sent": 0,
        "hardware_execution_ready": False,
        "diagnostic_only": diagnostic_only,
    }
    for name, value in [
        ("config.json", config),
        ("planning_config.json", limits),
        ("trial_profile.json", trial),
    ]:
        save_json(out / name, value)
    stage_b = None
    if args.stage_b_report:
        stage_b = json.loads(args.stage_b_report.read_text())
        report["stage_b_evidence_sha256"] = sha256(args.stage_b_report)
        save_json(out / "stage_b_evidence.json", stage_b)
    report["stage_d_blockers"] = hardware_trial_blockers(stage_b, hardware_trial) + (["explicit_diagnostic_only"] if diagnostic_only else [])
    report["stage_d_execution_status"] = "not_executed"
    observer = policy = None
    rejected = Counter()
    try:
        pre = json.loads(args.preflight_report.read_text())
        if (
            pre.get("status") != "passed"
            or pre["config_sha256"] != report["config_sha256"]
            or not 0 <= (time.time_ns() - pre["finished_unix_ns"]) / 1e9 < 3600
        ):
            raise ValueError("Fresh successful preflight for this config required")
        for path, digest in limits["solver_file_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("Solver/model changed: " + path)
        parameters = json.loads(
            (args.preflight_report.parent / "parameters.json").read_text()
        )["/tj/qp_controller"]
        for key, value in limits["expected_qp_parameters"].items():
            if parameters.get(key) != value:
                raise ValueError("QP parameter changed: " + key)
        solver = NativeQP(parameters, limits)
        sys.path.insert(0, config["repository"])
        from examples.forcevla.deployment.policy import ForceVLARemotePolicy

        policy = ForceVLARemotePolicy(
            config["address"], timeout=limits["rpc_timeout_s"]
        )
        if not policy.healthy():
            raise ValueError("Model service not healthy")
        sampling = dict(config, require_joints=True, require_joint_velocity=True)
        store = ObservationStore(sampling)
        observer = ROSObserver(sampling, store).start()
        deadline = time.monotonic() + args.duration
        while True:
            if observer.failure:
                raise RuntimeError(observer.failure)
            if time.monotonic() > deadline:
                raise TimeoutError("No fresh complete observation")
            try:
                request, metadata = store.snapshot()
                break
            except Rejected as exc:
                rejected[str(exc)] += 1
                time.sleep(0.03)
        observed = metadata["snapshot_monotonic_ns"] / 1e9 - max(
            v["source_age_s"] for v in metadata["streams"].values()
        )
        request_id = uuid.uuid4().hex
        request["seed"] = args.seed
        seed = np.asarray(metadata["joint_positions_rad"])
        velocity = np.asarray(metadata["streams"]["joints"]["velocities_rad_s"])
        initial_grip = np.asarray(
            metadata["streams"]["gripper"]["applied_target"], dtype=float
        )
        if (
            initial_grip.shape != (2,)
            or not np.isfinite(initial_grip).all()
            or np.any(initial_grip < 0)
            or np.any(initial_grip > 1)
        ):
            raise ValueError("Invalid current gripper targets")
        metadata["request_id"] = request_id
        metadata["original_observed_at"] = observed
        save_json(out / "observation_metadata.json", metadata)
        # Keep pre-RPC IO small; image snapshots and preview are saved after preparation.
        started = time.monotonic()
        result = predict_observation(policy, request, request_id)
        rpc_end = time.monotonic()
        report.update(
            request_id=request_id,
            rpc_ms=(rpc_end - started) * 1000,
            response_observation_age_s=rpc_end - observed,
        )
        bounded, adaptation = prepare_actions(
            result["actions"], limits["max_gripper_correction"]
        )
        save_json(out / "action_adaptation.json", adaptation)
        if bounded is None:
            raise ValueError("Gripper correction budget exceeded")
        action_check = check_actions(
            bounded, request["state"], rpc_end - observed, limits
        )
        save_json(out / "action_checks.json", action_check)
        if action_check["reasons"]:
            report["stage_c_status"] = "rejected_by_action_checks"
            report["rejection_reasons"] = action_check["reasons"]
            raise ValueError(
                "Action checks failed: " + ",".join(action_check["reasons"])
            )
        plan = prepare_native_plan(
            solver,
            actions=result["actions"],
            seed=seed,
            state=request["state"],
            request_id=request_id,
            observed_at=observed,
            output=out / "plan",
            prefix_steps=trial["prefix_steps"],
            max_correction=limits["max_gripper_correction"],
            max_age_s=limits["max_response_age_s"],
        )
        c_finished = time.monotonic()
        report.update(
            stage_c_status="prepared_fresh_prefix",
            stage_c_observation_age_s=c_finished - observed,
            prefix_steps=trial["prefix_steps"],
            prefix_ticks=len(plan.joints),
            dt=plan.dt,
        )
        trial_check = check_motion_prefix(
            plan,
            solver,
            trial,
            seed_velocity=velocity,
            initial_gripper_targets=initial_grip,
            diagnostic_only=diagnostic_only,
        )
        save_json(out / "stage_d_budget_checks.json", trial_check)
        report["stage_d_budget_reasons"] = trial_check["reasons"]
        report["stage_d_blockers"] = hardware_trial_blockers(stage_b, hardware_trial) + (["explicit_diagnostic_only"] if diagnostic_only else [])
        report["stage_d_execution_status"] = "not_executed"
        report["status"] = "prepared"
        np.savez_compressed(
            out / "prepared_prefix.npz",
            joints=plan.joints,
            grippers=plan.grippers,
            seed=plan.seed,
            dt=plan.dt,
            original_observed_at=plan.observed_at,
        )
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        if report.get("rejection_reasons"):
            report["status"] = "rejected"
        report["error"] = repr(exc)
    finally:
        if "request" in locals() and "velocity" in locals():
            np.savez_compressed(
                out / "observation.npz",
                **request,
                joint_positions_rad=seed,
                seed_velocity_rad_s=velocity,
            )
        if "result" in locals():
            np.savez_compressed(
                out / "prediction.npz",
                actions=result["actions"],
                normalized_actions=result["normalized_actions"],
            )
        if "request" in locals():
            grid = np.vstack(
                [
                    np.hstack([request[CAMERAS[0]], request[CAMERAS[1]]]),
                    np.hstack([request[CAMERAS[2]], request[CAMERAS[3]]]),
                ]
            )
            Image.fromarray(grid).save(out / "camera_preview.png")
        if observer:
            observer.close()
            report["diagnostics"] = store.diagnostics()
        if policy:
            policy.close()
        report.update(
            finished_unix_ns=time.time_ns(), rejected_observations=dict(rejected)
        )
        if "observed" in locals():
            report["age_at_report_s"] = time.monotonic() - observed
        save_json(out / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(out / "report.json"),
                "stage_d_budget_reasons": report.get("stage_d_budget_reasons"),
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "prepared" else 1


if __name__ == "__main__":
    raise SystemExit(main())
