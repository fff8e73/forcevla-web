"""Capture real observations only. Does not call inference or issue robot commands."""

import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

from robot_control.common import load_config, new_run, save_json, sha256, source_manifest, ROOT
from robot_control.observation import CAMERAS, ObservationStore, Rejected
from robot_control.ros_observer import ROSObserver


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "config/robot.json")
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--hz", type=float, default=2)
    parser.add_argument("--max-samples", type=int, default=20)
    args = parser.parse_args()
    if (
        not 0 < args.duration <= 3600
        or not 0 < args.hz <= 30
        or not 1 <= args.max_samples <= 10000
    ):
        parser.error("duration must be (0,3600], hz (0,30], max-samples [1,10000]")
    config = load_config(args.config)
    output = new_run(args.output, "observe")
    report = {
        "status": "failed",
        "scope": "read_only_observation_capture",
        "started_unix_ns": time.time_ns(),
        "config_sha256": sha256(args.config),
        "source_sha256": source_manifest(),
        "application_control_publishers": 0,
        "model_inference_requests": 0,
        "physical_validation_passed": False,
        "accepted_snapshots": 0,
        "sample_target": args.max_samples,
        "duration_limit_s": args.duration,
        "capture_hz": args.hz,
    }
    save_json(output / "config.json", config)
    store = ObservationStore(config)
    observer = ROSObserver(config, store)
    rejected, ages, skews = Counter(), [], []
    try:
        preflight = json.loads(args.preflight_report.read_text())
        if (
            preflight.get("status") != "passed"
            or preflight.get("config_sha256") != report["config_sha256"]
        ):
            raise ValueError("A passed preflight for this exact config is required")
        age = (time.time_ns() - preflight["finished_unix_ns"]) / 1e9
        if not 0 <= age <= 3600:
            raise ValueError("Preflight must be from the last hour")
        report["preflight_report"] = str(args.preflight_report.resolve())
        report["preflight_sha256"] = sha256(args.preflight_report)
        observer.start()
        started = time.monotonic()
        next_capture = started
        with (output / "observations.jsonl").open("w") as log:
            while (
                time.monotonic() - started < args.duration
                and report["accepted_snapshots"] < args.max_samples
            ):
                if observer.failure:
                    raise RuntimeError(observer.failure)
                if time.monotonic() < next_capture:
                    time.sleep(min(0.05, next_capture - time.monotonic()))
                    continue
                next_capture = time.monotonic() + 1 / args.hz
                try:
                    request, metadata = store.snapshot()
                except Rejected as exc:
                    rejected[str(exc)] += 1
                    continue
                index = report["accepted_snapshots"]
                name = f"observation_{index:06d}.npz"
                np.savez_compressed(output / name, **request)
                metadata.update(
                    observation_id=index,
                    file=name,
                    sha256=sha256(output / name),
                    state=request["state"].tolist(),
                    prompt=request["prompt"],
                )
                log.write(
                    json.dumps(metadata, ensure_ascii=False, allow_nan=False) + "\n"
                )
                log.flush()
                if index == 0:
                    grid = np.concatenate(
                        (
                            np.concatenate(
                                (request[CAMERAS[0]], request[CAMERAS[1]]), axis=1
                            ),
                            np.concatenate(
                                (request[CAMERAS[2]], request[CAMERAS[3]]), axis=1
                            ),
                        ),
                        axis=0,
                    )
                    Image.fromarray(grid).save(output / "camera_preview.png")
                report["accepted_snapshots"] += 1
                skews.append(metadata["cross_source_skew_s"])
                ages.append(
                    max(v["source_age_s"] for v in metadata["streams"].values())
                )
                print(
                    json.dumps(
                        {
                            "captured": index + 1,
                            "max_source_age_s": ages[-1],
                            "skew_s": skews[-1],
                        }
                    ),
                    flush=True,
                )
        if report["accepted_snapshots"] < args.max_samples:
            raise RuntimeError(
                "Sample target not reached before deadline; partial data retained, inspect diagnostics"
            )
        report["status"] = "passed_readonly_capture"
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
        report["max_source_age_s"] = max(ages)
        report["max_cross_source_skew_s"] = max(skews)
    except KeyboardInterrupt:
        report["status"] = "interrupted"
    except Exception as exc:
        report["error"] = repr(exc)
    finally:
        try:
            observer.close()
        except Exception as exc:
            report["status"] = "failed"
            report["cleanup_error"] = repr(exc)
        report.update(
            finished_unix_ns=time.time_ns(),
            rejected_snapshots=dict(rejected),
            diagnostics=store.diagnostics(),
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
    return 0 if report["status"] == "passed_readonly_capture" else 1


if __name__ == "__main__":
    raise SystemExit(main())
