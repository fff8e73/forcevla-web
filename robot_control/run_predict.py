"""Real observations -> existing RPC -> offline bounded IK. No control publishers."""

import argparse
from collections import Counter
import json
import math
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
from robot_control.prediction_checks import check_actions, check_analysis_age
from robot_control.model_rpc import predict_observation
from robot_control.action_adapter import prepare_actions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument(
        "--planning-config", type=Path, default=ROOT / "config/planning.json"
    )
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--predictions", type=int, default=3)
    parser.add_argument("--duration", type=float, default=90)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if (
        not 1 <= args.predictions <= 100
        or not 0 < args.duration <= 3600
        or args.seed < 0
    ):
        parser.error("Invalid bounded run options")
    config = load_config(args.config)
    limits = json.loads(args.planning_config.read_text())
    for key in [
        "max_response_age_s",
        "max_gripper_correction",
        "max_displacement_m",
        "max_rotation_rad",
        "max_tcp_step_m",
        "max_tcp_step_rad",
        "seed_position_tolerance_m",
        "seed_rotation_tolerance_rad",
        "tracking_position_tolerance_m",
        "tracking_rotation_tolerance_rad",
        "max_joint_step_rad",
        "solver_timeout_s",
        "rpc_timeout_s",
        "posture_regularization",
        "collision_probe_step_rad",
    ]:
        if (
            type(limits.get(key)) not in (float, int)
            or not math.isfinite(limits[key])
            or limits[key] <= 0
        ):
            parser.error("Invalid diagnostic budget: " + key)
    if type(limits.get("max_nfev")) is not int or not 1 <= limits["max_nfev"] <= 1000:
        parser.error("max_nfev must be an integer in [1,1000]")
    output = new_run(args.output, "prediction")
    report = {
        "status": "failed",
        "scope": "prediction_software_pipeline_not_hardware_acceptance",
        "started_unix_ns": time.time_ns(),
        "config_sha256": sha256(args.config),
        "planning_config_sha256": sha256(args.planning_config),
        "source_sha256": source_manifest(),
        "application_control_publishers": 0,
        "hardware_execution_ready": False,
        "attempted_predictions": 0,
        "completed_predictions": 0,
        "diagnostic_accepted_predictions": 0,
        "records": [],
        "operator_confirmation": config.get("operator_confirmation", {}),
    }
    save_json(output / "config.json", config)
    save_json(output / "planning_config.json", limits)
    observer = policy = None
    store = None
    rejected = Counter()
    try:
        pre = json.loads(args.preflight_report.read_text())
        if (
            pre.get("status") != "passed"
            or pre["config_sha256"] != report["config_sha256"]
        ):
            raise ValueError("Passed preflight for the same config required")
        if not 0 <= (time.time_ns() - pre["finished_unix_ns"]) / 1e9 <= 3600:
            raise ValueError("Preflight expired (>1h)")
        for path, digest in limits["solver_file_sha256"].items():
            if sha256(path) != digest:
                raise ValueError("Offline model/constraint artifact changed: " + path)
        parameters = json.loads(
            (args.preflight_report.parent / "parameters.json").read_text()
        )["/tj/qp_controller"]
        for key, value in limits["expected_qp_parameters"].items():
            if parameters.get(key) != value:
                raise ValueError("QP parameter changed: " + key)
        if limits.get("solver_backend") != "native_qp":
            raise ValueError(
                "Expected explicitly configured native_qp backend; no silent fallback"
            )
        solver = NativeQP(parameters, limits)
        save_json(output / "solver_manifest.json", solver.manifest)
        report["preflight_sha256"] = sha256(args.preflight_report)
        sys.path.insert(0, config["repository"])
        from examples.forcevla.deployment.policy import ForceVLARemotePolicy

        policy = ForceVLARemotePolicy(
            config["address"], timeout=limits["rpc_timeout_s"]
        )
        if not policy.healthy():
            raise RuntimeError("Inference service unhealthy")
        sampling = dict(config, require_joints=True)
        store = ObservationStore(sampling)
        observer = ROSObserver(sampling, store).start()
        start = time.monotonic()
        while (
            time.monotonic() - start < args.duration
            and report["attempted_predictions"] < args.predictions
        ):
            if observer.failure:
                raise RuntimeError(observer.failure)
            try:
                request, metadata = store.snapshot()
            except Rejected as exc:
                rejected[str(exc)] += 1
                time.sleep(0.05)
                continue
            number = report["attempted_predictions"]
            report["attempted_predictions"] += 1
            folder = output / f"prediction_{number:04d}"
            folder.mkdir()
            record = {
                "observation_id": uuid.uuid4().hex,
                "index": number,
                "status": "failed",
                "snapshot": metadata,
                "seed": args.seed,
                "hardware_execution_ready": False,
            }
            try:
                seed = np.asarray(metadata["joint_positions_rad"], dtype=np.float64)
                request["seed"] = args.seed
                np.savez_compressed(
                    folder / "observation.npz", **request, joint_positions_rad=seed
                )
                save_json(folder / "observation_metadata.json", metadata)
                if number == 0:
                    grid = np.vstack(
                        [
                            np.hstack([request[CAMERAS[0]], request[CAMERAS[1]]]),
                            np.hstack([request[CAMERAS[2]], request[CAMERAS[3]]]),
                        ]
                    )
                    Image.fromarray(grid).save(folder / "camera_preview.png")
                before_counts = store.diagnostics()["counters"]
                before = time.monotonic_ns()
                result = predict_observation(policy, request, record["observation_id"])
                after = time.monotonic_ns()
                # Include age of the oldest input at snapshot plus monotonic elapsed time.
                response_age = (
                    max(v["source_age_s"] for v in metadata["streams"].values())
                    + (after - before) / 1e9
                    + (before - metadata["snapshot_monotonic_ns"]) / 1e9
                )
                record.update(
                    rpc_ms=(after - before) / 1e6,
                    inference_ms=result["inference_time_us"] / 1000,
                    response_observation_age_s=response_age,
                    rpc_request_id_echo_validated=True,
                    receiver_updates_during_rpc={
                        key: count - before_counts.get(key, 0)
                        for key, count in store.diagnostics()["counters"].items()
                        if key.startswith("accepted:")
                    },
                )
                np.savez_compressed(
                    folder / "prediction.npz",
                    actions=result["actions"],
                    normalized_actions=result["normalized_actions"],
                )
                raw_action_report = check_actions(
                    result["actions"], request["state"], response_age, limits
                )
                save_json(folder / "raw_action_checks.json", raw_action_report)
                candidate, adaptation = prepare_actions(
                    result["actions"], limits["max_gripper_correction"]
                )
                save_json(folder / "action_adaptation.json", adaptation)
                record["raw_action_reasons"] = raw_action_report["reasons"]
                record["gripper_adaptation"] = {
                    k: adaptation[k]
                    for k in [
                        "status",
                        "changed_values",
                        "max_correction",
                        "max_allowed_correction",
                    ]
                }
                if candidate is None:
                    action_report = dict(raw_action_report)
                    action_report["reasons"] = sorted(
                        set(action_report["reasons"] + adaptation["reasons"])
                    )
                    ik = {
                        "status": "not_run",
                        "reasons": ["action_preparation_rejected"],
                        "environment_collision_checked": False,
                        "hardware_execution_ready": False,
                        "manifest": solver.manifest,
                    }
                else:
                    np.savez_compressed(
                        folder / "execution_candidate.npz", actions=candidate
                    )
                    action_report = check_actions(
                        candidate, request["state"], response_age, limits
                    )
                    ik = solver.rollout(candidate[0], seed, request["state"], folder)
                save_json(folder / "action_checks.json", action_report)
                save_json(folder / "ik_checks.json", ik)
                record["action_reasons"] = action_report["reasons"]
                record["ik_reasons"] = ik["reasons"]
                record["analysis_finished_age_s"] = (
                    response_age + (time.monotonic_ns() - after) / 1e9
                )
                record["post_analysis_reasons"] = check_analysis_age(
                    record["analysis_finished_age_s"], limits["max_response_age_s"]
                )
                record["status"] = (
                    "accepted_by_diagnostic_gates"
                    if not record["action_reasons"]
                    and not record["ik_reasons"]
                    and not record["post_analysis_reasons"]
                    else "rejected_by_diagnostic_gates"
                )
                report["completed_predictions"] += 1
                if record["status"] == "accepted_by_diagnostic_gates":
                    report["diagnostic_accepted_predictions"] += 1
                record["artifacts"] = {
                    p.name: sha256(p) for p in folder.iterdir() if p.is_file()
                }
            except Exception as exc:
                record["error"] = repr(exc)
            finally:
                save_json(folder / "report.json", record)
                report["records"].append(record)
                print(
                    json.dumps(
                        {
                            k: record.get(k)
                            for k in [
                                "index",
                                "status",
                                "rpc_ms",
                                "action_reasons",
                                "ik_reasons",
                                "post_analysis_reasons",
                                "error",
                            ]
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        if report["completed_predictions"] != args.predictions:
            raise RuntimeError(
                "Requested model predictions did not all complete; failed records retained"
            )
        completed = report["records"]
        report["summary"] = {
            "rpc_ms": {
                "median": float(np.median([x["rpc_ms"] for x in completed])),
                "max": max(x["rpc_ms"] for x in completed),
            },
            "response_age_s_max": max(
                x["response_observation_age_s"] for x in completed
            ),
            "gripper_adapted_values": sum(
                x["gripper_adaptation"]["changed_values"] for x in completed
            ),
            "max_gripper_correction": max(
                x["gripper_adaptation"]["max_correction"] for x in completed
            ),
            "action_rejection_counts": dict(
                Counter(r for x in completed for r in x["action_reasons"])
            ),
            "post_analysis_rejection_counts": dict(
                Counter(r for x in completed for r in x["post_analysis_reasons"])
            ),
            "analysis_age_s_max": max(x["analysis_finished_age_s"] for x in completed),
            "ik_rejection_counts": dict(
                Counter(r for x in completed for r in x["ik_reasons"])
            ),
        }
        report["status"] = "completed_prediction_run"
        report["diagnostic_assessment"] = (
            "all_passed"
            if report["diagnostic_accepted_predictions"] == args.predictions
            else "contains_rejections"
        )
        report["pending"] = [
            "physical camera mapping",
            "online/training pixel equivalence",
            "physical TCP and wrench semantics",
            "task object/environment collision geometry",
            "full online wrapper/mux/DAQP-revision equivalence",
            "ROS wrapper/mux/plant behavior",
            "representative task validation and approved thresholds",
        ]
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if observer:
            try:
                observer.close()
            except Exception as exc:
                report.update(status="failed", cleanup_error=repr(exc))
        if policy:
            policy.close()
        report.update(
            finished_unix_ns=time.time_ns(),
            rejected_observations=dict(rejected),
            diagnostics=store.diagnostics() if store else {},
        )
        save_json(output / "report.json", report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "report": str(output / "report.json"),
                "error": report.get("error"),
            },
            ensure_ascii=False,
        )
    )
    return 0 if report["status"] == "completed_prediction_run" else 1


if __name__ == "__main__":
    raise SystemExit(main())
