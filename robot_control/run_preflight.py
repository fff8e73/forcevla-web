"""Check the deployed service, local contracts and one frozen input; warm RPC only."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import shlex
import re
import sys
import time
import urllib.request

import numpy as np
import yaml

from robot_control.common import ROOT, load_config, new_run, save_json, sha256, source_manifest


def subset_differences(expected, actual, prefix=""):
    differences = []
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return [prefix + ": expected mapping"]
        for key, value in expected.items():
            if key not in actual:
                differences.append(prefix + "." + key + ": missing")
            else:
                differences.extend(
                    subset_differences(value, actual[key], prefix + "." + key)
                )
    elif expected != actual:
        differences.append(f"{prefix}: expected {expected!r}, actual {actual!r}")
    return differences


def parameter_snapshot(node):
    result = subprocess.run(
        ["ros2", "param", "dump", "--no-daemon", "--spin-time", "3", "--timeout", "5", node],
        capture_output=True, text=True, timeout=20
    )
    if result.returncode:
        raise RuntimeError(f"{node}: {result.stderr}")
    data = yaml.safe_load(result.stdout)
    return data[node]["ros__parameters"]


def numeric_metrics(actual, expected, thresholds):
    if (
        actual.shape != (1, 50, 32)
        or expected.shape != actual.shape
        or not np.isfinite(actual).all()
    ):
        raise ValueError("Invalid normalized prediction")
    x, y = actual[..., :14].astype(np.float64), expected[..., :14].astype(np.float64)
    error = x - y
    values = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "max_absolute_error": float(np.abs(error).max()),
        "relative_l2": float(
            np.linalg.norm(error)
            / max(np.linalg.norm(y), thresholds["relative_l2_denominator_floor"])
        ),
        "cosine": float(
            np.sum(x * y) / max(np.linalg.norm(x) * np.linalg.norm(y), 1e-12)
        ),
    }
    values["passed"] = bool(
        values["rmse"] <= thresholds["rmse_max"]
        and values["max_absolute_error"] <= thresholds["absolute_error_max"]
        and values["relative_l2"] <= thresholds["relative_l2_max"]
        and values["cosine"] > 0.99
    )
    return values


def systemd_status(service):
    result = subprocess.run(
        [
            "systemctl",
            "--user",
            "show",
            service,
            "-p",
            "MainPID",
            "-p",
            "ActiveState",
            "-p",
            "SubState",
            "-p",
            "ExecMainStartTimestamp",
        ],
        text=True,
        capture_output=True,
        check=True,
        timeout=5,
    )
    return dict(
        line.split("=", 1) for line in result.stdout.splitlines() if "=" in line
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warm-requests", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.warm_requests <= 10:
        parser.error("--warm-requests must be in [1,10]")
    config = load_config(args.config)
    output = new_run(args.output, "preflight")
    report = {
        "status": "failed",
        "scope": "service_preflight_one_frozen_sample_not_full_model_acceptance",
        "started_unix_ns": time.time_ns(),
        "config_sha256": sha256(args.config),
        "source_sha256": source_manifest(),
        "application_control_publishers": 0,
        "physical_validation_passed": False,
        "checks": {},
    }
    save_json(output / "config.json", config)
    policy = None
    try:
        repository, runtime = Path(config["repository"]), Path(config["runtime"])
        checkpoint = runtime / "converted-job4252-v1"
        status = systemd_status(config["service"])
        report["checks"]["service"] = status
        if status["ActiveState"] != "active" or int(status["MainPID"]) <= 0:
            raise RuntimeError(
                "Deployed service is not active; preflight never starts or restarts it"
            )
        pid = int(status["MainPID"])
        command = (
            Path(f"/proc/{pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
        )
        report["checks"]["process_title"] = command
        launch = subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                config["service"],
                "-p",
                "ExecStart",
                "--value",
            ],
            text=True,
            timeout=5,
        )
        report["checks"]["systemd_exec_start"] = launch
        # The engine uses setproctitle, so /proc/cmdline no longer carries its flags.
        command = shlex.split(launch)
        if (
            "examples.forcevla.serve" not in command
            or Path(f"/proc/{pid}/cwd").resolve() != repository
        ):
            raise RuntimeError("Unexpected service entry point or working directory")
        sockets = subprocess.check_output(
            ["ss", "-ltnp", "sport = :50071"], text=True, timeout=5
        )
        report["checks"]["listening_socket"] = sockets
        if not re.search(r"pid=" + str(pid) + r"[,)]", sockets):
            raise RuntimeError(
                "RPC listening socket does not belong to the systemd main process"
            )
        for flag, expected in [
            ("--backend", config["backend"]),
            ("--checkpoint", str(checkpoint)),
            ("--host", "127.0.0.1"),
            ("--port", "50071"),
            ("--manifest", str(runtime / "artifacts/deployed-runtime.json")),
        ]:
            if flag not in command or command[command.index(flag) + 1] != expected:
                raise RuntimeError("Unexpected service option: " + flag)
        if config["address"] != "127.0.0.1:50071":
            raise ValueError(
                "This local service provenance check supports only 127.0.0.1:50071"
            )
        accepted = json.loads(
            (runtime / "artifacts/service-025/manifest.json").read_text()
        )
        deployed = json.loads((runtime / "artifacts/deployed-runtime.json").read_text())
        acceptance_report = json.loads(
            (runtime / "artifacts/service-025/report.json").read_text()
        )
        if (
            acceptance_report["status"] != "passed"
            or acceptance_report["backend"] != config["backend"]
        ):
            raise ValueError("Saved service acceptance is not passed for this backend")
        if (
            accepted != deployed
            or accepted["sha256"] != acceptance_report["runtime_manifest"]
        ):
            raise ValueError("Deployed manifest differs from accepted runtime")
        print(
            "Checking deployed source, assets and complete weight SHA-256...",
            flush=True,
        )
        failures = []
        for section, base in [
            ("source_sha256", repository),
            ("asset_sha256", checkpoint),
        ]:
            for relative, expected in accepted[section].items():
                if sha256(base / relative) != expected:
                    failures.append(str(base / relative))
        if (
            sha256(checkpoint / "model.safetensors")
            != accepted["weight_file"]["sha256"]
        ):
            failures.append("model.safetensors")
        report["checks"]["runtime_manifest"] = {
            "sha256": accepted["sha256"],
            "mismatched_files": failures,
            "provenance_scope": "systemd ExecStart and socket owner, saved manifest, current disk hashes and frozen output; process title hides argv and RPC exposes no live manifest API",
        }
        if failures:
            raise ValueError("Accepted source/assets/weights changed")
        contract_failures = [
            path
            for path, expected in config["expected_file_sha256"].items()
            if sha256(path) != expected
        ]
        report["checks"]["model_files"] = {"mismatched_files": contract_failures}
        if contract_failures:
            raise ValueError("Hardware model/conversion files changed")
        print(
            "Reading ROS parameters and existing HTTP status (no control calls)...",
            flush=True,
        )
        parameters, differences = {}, []
        nodes = list(config["expected_parameters"])
        with ThreadPoolExecutor(max_workers=5) as pool:
            futures = {node: pool.submit(parameter_snapshot, node) for node in nodes}
            for node, future in futures.items():
                try:
                    parameters[node] = future.result()
                    differences.extend(
                        subset_differences(
                            config["expected_parameters"][node], parameters[node], node
                        )
                    )
                except Exception as exc:
                    parameters[node] = {"error": repr(exc)}
                    differences.append(node + ": query failed")
        save_json(output / "parameters.json", parameters)
        report["checks"]["runtime_parameters"] = {"differences": differences}
        statuses = {}
        for name, url in config["http_status"].items():
            with urllib.request.urlopen(url, timeout=5) as response:
                statuses[name] = json.load(response)
            if statuses[name].get("ok") is not True:
                differences.append(name + ": status unavailable")
        report["checks"]["robot_and_gripper_status"] = statuses
        gripper = statuses["gripper"]["data"]
        if gripper.get("calibration") != config["expected_calibration"]:
            differences.append("live gripper calibration differs")
        if differences:
            raise ValueError(
                "Runtime contract changed; inspect parameters/status and revise with evidence"
            )
        sys.path.insert(0, str(repository))
        from examples.forcevla.deployment.policy import ForceVLARemotePolicy
        from examples.forcevla.transport.codec import recorded_observation
        from examples.forcevla.validation.frozen_reference import (
            verify_frozen_reference,
        )

        acceptance_path = repository / "examples/forcevla/validation/acceptance.json"
        reference = runtime / "baseline-job5444"
        report["checks"]["frozen_reference"] = verify_frozen_reference(
            reference, acceptance_path
        )
        criteria = json.loads(acceptance_path.read_text())[
            "normalized_action_per_sample"
        ]
        with np.load(reference / "sample_000.npz", allow_pickle=False) as sample:
            request = recorded_observation(sample)
            expected_original = sample["normalized_actions"].copy()
        with np.load(
            runtime / "artifacts/service-025/direct/sample_000.npz", allow_pickle=False
        ) as sample:
            expected_deployed = {
                k: sample[k].copy() for k in ["actions", "normalized_actions"]
            }
        report["checks"]["gpu_context"] = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        ).stdout
        policy = ForceVLARemotePolicy(config["address"], timeout=120)
        if not policy.healthy():
            raise RuntimeError("gRPC model health failed")
        report["checks"]["grpc_health"] = True
        timings, model_ms = [], []
        for index in range(args.warm_requests + 1):
            before = time.perf_counter()
            result = policy.predict(request)
            timings.append((time.perf_counter() - before) * 1000)
            model_ms.append(result["inference_time_us"] / 1000)
            metrics = numeric_metrics(
                result["normalized_actions"], expected_original, criteria
            )
            parity = all(
                np.allclose(result[k], v, atol=1e-6, rtol=0)
                for k, v in expected_deployed.items()
            )
            if not metrics["passed"] or not parity:
                raise ValueError(
                    f"Frozen sample failed: original={metrics}, accepted_parity={parity}"
                )
            if index == 0:
                np.savez_compressed(
                    output / "frozen_prediction.npz",
                    actions=result["actions"],
                    normalized_actions=result["normalized_actions"],
                )
                report["checks"]["frozen_prediction"] = {
                    "original_metrics": metrics,
                    "accepted_parity_atol_1e6": parity,
                    "sample": "sample_000.npz",
                    "output_shapes": {
                        k: list(result[k].shape) for k in expected_deployed
                    },
                }
            print(
                f"Frozen RPC {index + 1}: {timings[-1]:.2f} ms, parity passed",
                flush=True,
            )
        report["timing"] = {
            "first_request_this_run_ms": timings[0],
            "warm_requests_ms": timings[1:],
            "model_execution_ms": model_ms,
            "warm_median_ms": float(np.median(timings[1:])),
            "note": "Already-running service; first request is not a measured cold start. This is a short readiness check, not a benchmark.",
        }
        if systemd_status(config["service"])["MainPID"] != str(pid):
            raise RuntimeError("Service restarted during verification")
        report["status"] = "passed"
        report["field_verification_pending"] = [
            k
            for k in [
                "camera_mapping_verified",
                "camera_transport_equivalence_verified",
                "physical_tcp_verified",
                "wrench_physical_semantics_verified",
            ]
            if not config[k]
        ]
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        if policy:
            policy.close()
        report["finished_unix_ns"] = time.time_ns()
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
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
