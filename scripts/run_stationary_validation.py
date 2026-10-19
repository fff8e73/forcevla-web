"""Explicit operator-requested stationary validation via the existing web workflow.

No new target or mode API is called directly. The web workflow owns validation,
Input=None selection and claim/hold/release. Private session is retained for
recovery if the workflow reports held/uncertain; it is never printed.
"""

import argparse
import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import httpx
import websockets


async def run(args):
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=False)
    base = "http://127.0.0.1:8096"
    report = {
        "started_unix": time.time(),
        "kind": args.kind,
        "scope": "operator requested stationary test, no new target",
        "safety_ack_supplied": args.safety_ack,
        "events": [],
    }
    async with httpx.AsyncClient(base_url=base, timeout=8) as client:
        session = (await client.get("/api/v1/session")).json()
        client.headers.update({"Origin": base, "X-CSRF-Token": session["csrf_token"]})
        cookie = client.cookies.get("forcevla_session")
        credential = root / "recovery-session.json"
        fd = os.open(credential, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"cookie": cookie, "csrf_token": session["csrf_token"]}, f)
        async with websockets.connect(
            "ws://127.0.0.1:8096/api/v1/events",
            origin=base,
            additional_headers={"Cookie": "forcevla_session=" + cookie},
            proxy=None,
        ) as ws:
            stop = asyncio.Event()

            async def keepalive():
                while not stop.is_set():
                    await ws.send("ping")
                    try:
                        await asyncio.wait_for(ws.recv(), timeout=2)
                    except TimeoutError:
                        pass
                    await asyncio.sleep(0.3)

            task = asyncio.create_task(keepalive())
            try:
                response = await client.post("/api/v1/control-validation/check")
                response.raise_for_status()
                report["precheck"] = response.json()
                if not report["precheck"]["ready"]:
                    report["status"] = "blocked_before_start"
                    return report
                response = await client.post(
                    "/api/v1/control-validation/start",
                    json={
                        "kind": args.kind,
                        "command_id": uuid.uuid4().hex,
                        "safety_ack": args.safety_ack,
                    },
                )
                response.raise_for_status()
                report["start"] = response.json()
                seen = None
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    response = await client.get("/api/v1/control-validation/status")
                    response.raise_for_status()
                    status = response.json()
                    current = status.get("run") or {}
                    if current.get("state") != seen:
                        seen = current.get("state")
                        print(
                            json.dumps(
                                {
                                    "state": seen,
                                    "reason": current.get("reason"),
                                    "post_count": current.get("post_count"),
                                },
                                ensure_ascii=False,
                            ),
                            flush=True,
                        )
                    report["latest"] = status
                    if seen not in {
                        "CHECKING",
                        "SELECTING_NONE",
                        "CLAIMING",
                        "OBSERVING",
                        "HOLDING",
                        "RELEASING",
                        "ABORTING",
                    }:
                        report["status"] = seen
                        if seen in {
                            "COMPLETED_STATIONARY",
                            "BLOCKED",
                            "ABORTED_UNOWNED",
                            "RELEASED",
                        }:
                            credential.unlink(missing_ok=True)
                        return report
                    await asyncio.sleep(0.15)
                await client.post("/api/v1/control-validation/abort")
                report["status"] = "timeout_abort_requested"
                return report
            finally:
                stop.set()
                await task


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--kind", choices=["stationary_cycle", "lease_expiry"], default="stationary_cycle"
    )
    p.add_argument("--safety-ack", action="store_true", required=True)
    args = p.parse_args()
    report = {}
    try:
        report = asyncio.run(run(args))
    except Exception as exc:
        report = {
            "status": "client_error",
            "error": repr(exc),
            "note": "Check web state; do not blindly repeat claim",
        }
    path = Path(args.output) / "report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"status": report.get("status"), "report": str(path)}, ensure_ascii=False))
    return 0 if report.get("status") == "COMPLETED_STATIONARY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
