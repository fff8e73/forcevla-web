"""Fault injection affects only this project's collector and inference session."""

import asyncio
import contextlib
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path

import httpx
from websockets.asyncio.client import connect


async def main():
    root = Path("artifacts/stability-20261014")
    root.mkdir(parents=True, exist_ok=True)
    main_pid = int(
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
    collector = int(subprocess.check_output(["pgrep", "-P", str(main_pid)], text=True).split()[0])
    cmd = Path(f"/proc/{collector}/cmdline").read_bytes()
    assert b"forcevla_web.collector" in cmd and b"forcevla-web" in cmd
    report = {"collector_pid": collector, "hardware_write": False}
    paused = False
    run_id = None
    heartbeat = None
    async with httpx.AsyncClient(
        base_url="http://127.0.0.1:8096", timeout=20, trust_env=False
    ) as client:
        session = (await client.get("/api/v1/session")).json()
        client.headers.update(
            {"Origin": "http://127.0.0.1:8096", "X-CSRF-Token": session["csrf_token"]}
        )
        cookie = "; ".join(f"{k}={v}" for k, v in client.cookies.items())
        ws = await connect(
            "ws://127.0.0.1:8096/api/v1/events",
            origin="http://127.0.0.1:8096",
            additional_headers={"Cookie": cookie},
        )

        async def beats():
            while True:
                await ws.send("ping")
                await asyncio.sleep(1)

        heartbeat = asyncio.create_task(beats())

        async def drain():
            async for _ in ws:
                pass

        receiver = asyncio.create_task(drain())

        async def status():
            return (await client.get("/api/v1/status")).json()

        async def wait(predicate, seconds=20):
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                s = await status()
                if predicate(s):
                    return s
                await asyncio.sleep(0.1)
            raise AssertionError("fault test condition timed out")

        try:
            await wait(lambda s: s["input"]["ready"] and s["model"]["healthy"])
            response = await client.post(
                "/api/v1/runs",
                json={
                    "kind": "continuous",
                    "prompt": session["default_prompt"],
                    "seed": 42,
                    "command_id": uuid.uuid4().hex,
                },
            )
            response.raise_for_status()
            run_id = response.json()["id"]
            await wait(lambda s: s.get("run", {}).get("count", 0) >= 2)
            os.kill(collector, signal.SIGSTOP)
            paused = True
            fault_start = time.monotonic()
            waiting = await wait(
                lambda s: not s["input"]["ready"] and s["run"]["state"] == "WAITING_INPUT", 8
            )
            count = waiting["run"]["count"]
            await asyncio.sleep(1)
            stopped = await status()
            assert stopped["run"]["count"] == count
            report["stale_input"] = {
                "waiting_after_s": time.monotonic() - fault_start,
                "reasons": stopped["input"]["reasons"],
                "no_new_predictions": True,
            }
            os.kill(collector, signal.SIGCONT)
            paused = False
            recovered = await wait(lambda s: s["input"]["ready"] and s["run"]["count"] > count, 12)
            report["recovered_count"] = recovered["run"]["count"]
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
            await ws.close()
            disconnect = time.monotonic()
            idle = await wait(lambda s: s["run"]["state"] == "IDLE", 20)
            report["owner_disconnect"] = {
                "settled_s": time.monotonic() - disconnect,
                "state": idle["run"]["state"],
            }
            events = (await client.get(f"/api/v1/runs/{run_id}/events")).json()
            assert any(e["payload"].get("reason") == "会话持有者断连" for e in events)
            report["passed"] = True
        finally:
            if paused:
                os.kill(collector, signal.SIGCONT)
            if run_id:
                with contextlib.suppress(Exception):
                    await client.post(f"/api/v1/runs/{run_id}/stop")
            if heartbeat:
                heartbeat.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await heartbeat
            await ws.close()
            receiver.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await receiver
            (root / "fault-recovery.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n"
            )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
