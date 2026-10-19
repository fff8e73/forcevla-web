"""Unprivileged atomic sink for the read-only controller telemetry helper."""

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid


def validate(row, now):
    if (
        row.get("source") != "passive_controller_sdk_feedback"
        or row.get("controller_ip") != "6.6.7.190"
        or row.get("commands_sent") != 0
    ):
        raise ValueError("Unexpected telemetry source")
    stamp = row.get("received_mono")
    if (
        type(stamp) not in (int, float)
        or not math.isfinite(stamp)
        or not 0 <= now - stamp <= 0.5
    ):
        raise ValueError("Stale or invalid source receipt")
    for name in ["target_rad", "positions_rad", "velocities_rad_s"]:
        value = row.get(name)
        if (
            not isinstance(value, list)
            or len(value) != 14
            or any(type(x) not in (int, float) or not math.isfinite(x) for x in value)
        ):
            raise ValueError("Invalid " + name)
    frames = row.get("frames")
    if (
        not isinstance(frames, list)
        or len(frames) != 2
        or any(type(x) is not int or not 0 <= x <= 1000000 for x in frames)
    ):
        raise ValueError("Invalid frame serials")
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("/dev/shm/forcevla-controller-readback/current.json"),
    )
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    session = uuid.uuid4().hex
    count = 0
    rejected = 0
    last = -1.0
    last_print = 0.0
    for line in sys.stdin:
        try:
            if len(line) > 65536:
                raise ValueError("Oversized line")
            row = validate(json.loads(line), time.monotonic())
            if row["received_mono"] <= last:
                raise ValueError("Duplicate/out-of-order receipt")
            last = row["received_mono"]
            row["readback_session"] = session
            row["stored_mono"] = time.monotonic()
            temp = args.output.with_name(args.output.name + "." + session + ".tmp")
            fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w") as stream:
                json.dump(row, stream, allow_nan=False)
            temp.replace(args.output)
            count += 1
            if time.monotonic() - last_print >= 5:
                print(
                    json.dumps(
                        {
                            "saved_frames": count,
                            "rejected": rejected,
                            "path": str(args.output),
                            "commands_sent": 0,
                        }
                    ),
                    flush=True,
                )
                last_print = time.monotonic()
        except (ValueError, TypeError, KeyError) as exc:
            rejected += 1
            if rejected <= 3:
                print("Rejected telemetry: " + str(exc), file=sys.stderr)
    print(
        json.dumps(
            {
                "reader_finished": True,
                "saved_frames": count,
                "rejected": rejected,
                "commands_sent": 0,
            }
        ),
        flush=True,
    )
    return 0 if count else 1


if __name__ == "__main__":
    raise SystemExit(main())
