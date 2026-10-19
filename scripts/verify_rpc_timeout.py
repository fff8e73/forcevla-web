"""Explicit live-RPC deadline test with synthetic observations and an isolated app."""

import argparse
import json
import tempfile
import time
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from forcevla_web.app import create_app
from forcevla_web.manager import Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import RemotePolicy
from forcevla_web.store import Store


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-model", action="store_true", required=True)
    parser.parse_args()
    config = json.loads(Path("config/local.json").read_text())
    config.update(source="demo", allowed_hosts=["testserver"], rpc_timeout_s=0.000001)
    with tempfile.TemporaryDirectory(prefix="forcevla-rpc-fault-") as root:
        config["data_dir"] = root
        policy = RemotePolicy(config)
        store = Store(root, 100)
        manager = Manager(config, DemoSource(), policy, store)
        with TestClient(create_app(config, manager)) as client:
            session = client.get("/api/v1/session").json()
            client.headers.update(
                {"Origin": "http://testserver", "X-CSRF-Token": session["csrf_token"]}
            )
            end = time.monotonic() + 10
            while time.monotonic() < end:
                state = client.get("/api/v1/status").json()
                if state["model"]["healthy"] and state["input"]["ready"]:
                    break
                time.sleep(0.02)
            response = client.post(
                "/api/v1/runs",
                json={
                    "kind": "continuous",
                    "prompt": config["default_prompt"],
                    "seed": 42,
                    "command_id": uuid.uuid4().hex,
                },
            )
            response.raise_for_status()
            run_id = response.json()["id"]
            end = time.monotonic() + 5
            while time.monotonic() < end and manager.run["state"] != "ERROR":
                time.sleep(0.02)
            assert manager.run["state"] == "ERROR"
            assert "DEADLINE_EXCEEDED" in manager.run["reason"]
            time.sleep(1)
            events = store.events(run_id)
            request_count = sum(e["type"] == "request" for e in events)
            assert request_count == 1
            healthy_after = policy.healthy()
            assert healthy_after
            report = {
                "source": "synthetic observation / real RPC / isolated web app",
                "rpc_timeout_s": config["rpc_timeout_s"],
                "request_count": request_count,
                "final_state": manager.run["state"],
                "error": manager.run["reason"],
                "model_healthy_after": healthy_after,
                "automatic_retry": False,
                "robot_motion_commands": False,
                "passed": True,
            }
    Path("artifacts/stability-20261014/rpc-timeout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
