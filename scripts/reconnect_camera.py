"""Explicit camera-only reconnect after a network-context change.

Without --restart-camera-if-idle this tool only reports status. It never changes
robot mode, input selection, gripper state, or recording sessions.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path

BASE = "http://127.0.0.1:8080"


def read(path):
    with urllib.request.urlopen(BASE + path, timeout=5) as response:
        result = json.load(response)
    if not result.get("ok"):
        raise RuntimeError("Backend did not confirm status")
    return result["data"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--restart-camera-if-idle", action="store_true")
    args = parser.parse_args()
    record = read("/api/v1/record/status")
    robot = read("/api/v1/robot/status")
    camera = read("/api/v1/modules/camera/status")
    report = {
        "time_unix": time.time(),
        "recording_active": record["is_recording"],
        "robot_input_mode": robot["current_input_mode"],
        "camera": camera,
        "action": "status_only",
    }
    if args.restart_camera_if_idle:
        if record["is_recording"] or record.get("phase") in (
            "starting",
            "recording",
            "stopping",
            "saving",
        ):
            raise SystemExit("Recording is active or transitioning; camera restart was not sent")
        if robot["current_input_mode"] != 0:
            raise SystemExit("Robot input is not None; camera restart was not sent")
        request = urllib.request.Request(
            BASE + "/api/v1/modules/camera/restart", data=b"", method="POST"
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            reply = json.load(response)
        report.update(action="camera_restart", response=reply, robot_motion_commands=False)
        root = Path("artifacts/camera-reconnect")
        root.mkdir(parents=True, exist_ok=True)
        (root / f"{int(time.time())}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        )
        if not reply.get("ok"):
            raise SystemExit("Camera restart was not confirmed; inspect its status before retrying")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
