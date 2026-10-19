"""Exercise only prediction HTTP APIs, with owner heartbeat and bounded run length."""

import argparse
import asyncio
import contextlib
import json
import time
import uuid
from pathlib import Path

import httpx
import numpy as np
from websockets.asyncio.client import connect


async def verify(base, samples, output):
    run_id = None
    async with httpx.AsyncClient(base_url=base, timeout=20, trust_env=False) as client:
        session = (await client.get("/api/v1/session")).json()
        client.headers.update({"Origin": base, "X-CSRF-Token": session["csrf_token"]})
        cookie = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        async with connect(
            base.replace("http", "ws", 1) + "/api/v1/events",
            origin=base,
            additional_headers={"Cookie": cookie},
        ) as socket:

            async def heartbeat():
                while True:
                    await socket.send("ping")
                    await asyncio.sleep(2)

            beat = asyncio.create_task(heartbeat())
            start = time.monotonic()
            stop_sent = False
            try:
                async with asyncio.timeout(180):
                    async for text in socket:
                        state = json.loads(text)
                        if run_id is None:
                            if not state["model"]["healthy"] or not state["input"]["ready"]:
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
                            run_id = response.json()["id"]
                        run = state.get("run")
                        if not run or run["id"] != run_id:
                            continue
                        if run["state"] == "ERROR":
                            raise RuntimeError(run["reason"])
                        if run["count"] >= samples and not stop_sent:
                            (await client.post(f"/api/v1/runs/{run_id}/stop")).raise_for_status()
                            stop_sent = True
                        if stop_sent and run["state"] == "IDLE":
                            break
                detail = (await client.get(f"/api/v1/runs/{run_id}")).json()
                valid = [p for p in detail["predictions"] if not p["discarded"]]
                steady = [p["rpc_s"] for p in valid if not p["startup_sample"]]
                report = {
                    "run_id": run_id,
                    "duration_s": time.monotonic() - start,
                    "completed": len(valid),
                    "discarded": sum(p["discarded"] for p in detail["predictions"]),
                    "rpc_steady_p50_s": float(np.median(steady)),
                    "rpc_steady_p95_s": float(np.percentile(steady, 95)),
                    "observation_age_max_s": max(p["observation_age_s"] for p in valid),
                    "image_contract_verified": False,
                    "hardware_write": False,
                    "state": detail["state"],
                    "source": detail["source"],
                }
                Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
                print(json.dumps(report, ensure_ascii=False, indent=2))
                assert len(valid) >= samples
            finally:
                if run_id:
                    await client.post(f"/api/v1/runs/{run_id}/stop")
                beat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await beat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="http://127.0.0.1:8096")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--output", default="artifacts/live-smoke-report.json")
    args = parser.parse_args()
    asyncio.run(verify(args.base, args.samples, args.output))
