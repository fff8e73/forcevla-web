"""Timed real prediction inference acceptance with a durable, predeclared gate file."""

import argparse
import asyncio
import contextlib
import json
import subprocess
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from websockets.asyncio.client import connect


def process_stats(pid):
    root = Path(f"/proc/{pid}")
    status = {
        line.split(":", 1)[0]: line.split(":", 1)[1].strip()
        for line in (root / "status").read_text().splitlines()
        if ":" in line
    }
    return {
        "pid": pid,
        "rss_mb": int(status["VmRSS"].split()[0]) / 1024,
        "fds": len(list((root / "fd").iterdir())),
    }


async def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    criteria = {
        "duration_s": args.duration,
        "input_valid_fraction_min": 0.99,
        "completed_min": int(args.duration * 0.5),
        "rpc_p95_s_max": 3.0,
        "error_events_max": 0,
        "rss_growth_mb_max": 100.0,
        "rss_peak_mb_max": 800.0,
        "fd_growth_max": 10,
        "stop_timeout_s": 20.0,
        "warmup_exclusion_s": 10.0,
    }
    (root / "criteria.json").write_text(json.dumps(criteria, indent=2) + "\n")
    pid = int(
        subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                "forcevla-web.service",
                "-p",
                "MainPID",
                "--value",
            ],
            text=True,
        )
    )
    children = []
    for _ in range(100):
        result = subprocess.run(["pgrep", "-P", str(pid)], text=True, capture_output=True)
        children = [int(p) for p in result.stdout.split()]
        if children:
            break
        await asyncio.sleep(0.1)
    if not children:
        raise RuntimeError("Collector did not start within 10 seconds")
    pids = [pid, *children]
    started = time.monotonic()
    run_started = None
    run_id = None
    last_pred = None
    last_sample = 0.0
    last_progress = 0.0
    metrics = []
    resources = []
    samples = []
    model_times = []
    prediction_ages = []
    errors = []
    stop_sent = None
    final = None
    async with httpx.AsyncClient(base_url=args.base, timeout=25, trust_env=False) as client:
        session = (await client.get("/api/v1/session")).json()
        client.headers.update({"Origin": args.base, "X-CSRF-Token": session["csrf_token"]})
        cookie = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        async with connect(
            args.base.replace("http", "ws", 1) + "/api/v1/events",
            origin=args.base,
            additional_headers={"Cookie": cookie},
            max_queue=4,
        ) as ws:

            async def heartbeat():
                while True:
                    await ws.send("ping")
                    await asyncio.sleep(2)

            beat = asyncio.create_task(heartbeat())
            try:
                async with asyncio.timeout(args.duration + 90):
                    with (root / "samples.jsonl").open("w") as stream:
                        async for text in ws:
                            state = json.loads(text)
                            now = time.monotonic()
                            if run_id is None:
                                if not state["model"]["healthy"] or not state["input"]["ready"]:
                                    if now - started > 30:
                                        raise RuntimeError(
                                            "Inputs did not become ready before the start budget"
                                        )
                                    continue
                                response = await client.post(
                                    "/api/v1/runs",
                                    json={
                                        "kind": "continuous",
                                        "prompt": session["default_prompt"],
                                        "seed": 42,
                                        "command_id": uuid.uuid4().hex,
                                    },
                                )
                                if response.status_code == 409 and "输入未就绪" in response.text:
                                    continue
                                response.raise_for_status()
                                created = response.json()
                                run_id = created["id"]
                                run_started = time.monotonic()
                                (root / "run.json").write_text(
                                    json.dumps(created, ensure_ascii=False, indent=2) + "\n"
                                )
                                print(
                                    f"RUN {run_id} started: {args.duration}s real prediction inference",
                                    flush=True,
                                )
                                continue
                            current = state.get("run")
                            if not current or current["id"] != run_id:
                                raise RuntimeError("Active run unexpectedly changed")
                            elapsed = now - run_started
                            if current["state"] == "ERROR":
                                raise RuntimeError(current["reason"])
                            if elapsed >= 10 and stop_sent is None:
                                samples.append(state["input"]["ready"])
                            prediction_id = state.get("latest_prediction_id")
                            if prediction_id and prediction_id != last_pred:
                                prediction = (
                                    await client.get(f"/api/v1/predictions/{prediction_id}")
                                ).json()
                                last_pred = prediction_id
                                if not prediction["startup_sample"]:
                                    metrics.append(prediction["rpc_s"])
                                    model_times.append(prediction["model_s"])
                                prediction_ages.append(prediction["observation_age_s"])
                            if now - last_sample >= 2:
                                processes = [process_stats(p) for p in pids]
                                resource = {
                                    "elapsed_s": elapsed,
                                    "rss_mb": sum(p["rss_mb"] for p in processes),
                                    "fds": sum(p["fds"] for p in processes),
                                    "processes": processes,
                                }
                                resources.append(resource)
                                record = {
                                    "elapsed_s": elapsed,
                                    "state": current["state"],
                                    "count": current["count"],
                                    "input_ready": state["input"]["ready"],
                                    "reasons": state["input"]["reasons"],
                                    "ages": state["input"]["ages"],
                                    "source_skew_s": state["input"].get("source_skew_s"),
                                    "transport": state["input"].get("transport"),
                                    **resource,
                                }
                                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                                stream.flush()
                                last_sample = now
                            if now - last_progress >= 60:
                                progress = {
                                    "elapsed_s": round(elapsed, 1),
                                    "count": current["count"],
                                    "input_valid_fraction": sum(samples) / len(samples)
                                    if samples
                                    else None,
                                    "rpc_p95_s": float(np.percentile(metrics, 95))
                                    if metrics
                                    else None,
                                    "rss_mb": resources[-1]["rss_mb"] if resources else None,
                                }
                                (root / "progress.json").write_text(
                                    json.dumps(progress, indent=2) + "\n"
                                )
                                print(json.dumps(progress), flush=True)
                                last_progress = now
                            if elapsed >= args.duration and stop_sent is None:
                                (
                                    await client.post(f"/api/v1/runs/{run_id}/stop")
                                ).raise_for_status()
                                stop_sent = time.monotonic()
                            if stop_sent is not None and current["state"] == "IDLE":
                                final = current
                                break
                            if stop_sent is not None and time.monotonic() - stop_sent > 20:
                                raise RuntimeError("Stop did not settle within 20 seconds")
            except asyncio.CancelledError:
                errors.append("Acceptance run interrupted before completion")
                raise
            except Exception as exc:
                errors.append(str(exc))
                raise
            finally:
                if run_id:
                    with contextlib.suppress(Exception):
                        await client.post(f"/api/v1/runs/{run_id}/stop")
                beat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await beat
                baseline = next(
                    (r for r in resources if r["elapsed_s"] >= 60),
                    resources[0] if resources else {"rss_mb": 0, "fds": 0},
                )
                peak = max((r["rss_mb"] for r in resources), default=0)
                growth = max(
                    (r["rss_mb"] - baseline["rss_mb"] for r in resources if r["elapsed_s"] >= 60),
                    default=0,
                )
                fd_growth = max(
                    (r["fds"] - baseline["fds"] for r in resources if r["elapsed_s"] >= 60),
                    default=0,
                )
                summary = {
                    "run_id": run_id,
                    "duration_s": time.monotonic() - run_started if run_started else 0,
                    "final": final,
                    "completed": final["count"] if final else (current["count"] if run_id else 0),
                    "input_valid_fraction": sum(samples) / len(samples) if samples else 0,
                    "input_samples": len(samples),
                    "rpc_samples": len(metrics),
                    "rpc_p50_s": float(np.percentile(metrics, 50)) if metrics else None,
                    "rpc_p95_s": float(np.percentile(metrics, 95)) if metrics else None,
                    "rpc_max_s": max(metrics, default=0),
                    "observation_age_max_s": max(prediction_ages, default=0),
                    "rss_peak_mb": peak,
                    "rss_growth_mb": growth,
                    "fd_growth": fd_growth,
                    "errors": errors,
                    "stop_settle_s": time.monotonic() - stop_sent if stop_sent else None,
                    "hardware_write": False,
                }
                summary["passed"] = bool(
                    final
                    and final["count"] >= criteria["completed_min"]
                    and summary["duration_s"] >= args.duration
                    and summary["input_valid_fraction"] >= 0.99
                    and summary["rpc_p95_s"] is not None
                    and summary["rpc_p95_s"] <= 3
                    and not errors
                    and peak <= 800
                    and growth <= 100
                    and fd_growth <= 10
                    and summary["stop_settle_s"] <= 20
                )
                (root / "report.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
                )
                print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8096")
    parser.add_argument("--duration", type=float, default=1800)
    parser.add_argument("--output", default="artifacts/stability-20261014/endurance")
    args = parser.parse_args()
    asyncio.run(run(args))
