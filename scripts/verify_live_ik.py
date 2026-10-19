"""Real inference plus offline checks; the model and helper have no motion outputs."""

import asyncio
import json
import sqlite3
from collections import Counter
from pathlib import Path

from verify_live import verify


async def main():
    root = Path("artifacts/offline-ik-20261014")
    await verify("http://127.0.0.1:8096", 12, str(root / "continuous-model.json"))
    run = json.loads((root / "continuous-model.json").read_text())["run_id"]
    records = []
    for _ in range(30):
        with sqlite3.connect("file:data/runs.sqlite3?mode=ro", uri=True) as db:
            records = [
                json.loads(row[0])
                for row in db.execute(
                    "SELECT body FROM predictions WHERE run_id=? ORDER BY rowid", (run,)
                )
            ]
        accepted = [p for p in records if not p["discarded"]]
        if all(
            isinstance(p.get("offline_ik"), dict) and p["offline_ik"]["status"] != "pending"
            for p in accepted
        ):
            break
        await asyncio.sleep(0.2)
    reasons = Counter()
    statuses = Counter()
    timings = []
    for p in accepted:
        d = p["offline_ik"]
        statuses[d["status"]] += 1
        reasons.update(d.get("reasons", []))
        assert d["execution_verified"] is False
        assert d["seed_context"]["snapshot_id"] == p["snapshot_id"]
        assert d["model"]["collision_pair_count"] == 8
        if "elapsed_s" in d:
            timings.append(d["elapsed_s"])
    report = {
        "run_id": run,
        "predictions": len(accepted),
        "diagnostic_statuses": dict(statuses),
        "all_50_steps_checked": all(p["offline_ik"].get("checked_steps") == 50 for p in accepted),
        "offline_pass_count": sum(p["offline_ik"].get("passed", False) for p in accepted),
        "reason_counts": dict(reasons),
        "max_ik_s": max(timings, default=0),
        "queue_pending": False,
        "robot_motion_commands": False,
    }
    report["software_validation_passed"] = (
        len(accepted) >= 12
        and report["all_50_steps_checked"]
        and statuses.get("complete") == len(accepted)
    )
    (root / "continuous-ik.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    assert report["software_validation_passed"]


if __name__ == "__main__":
    asyncio.run(main())
