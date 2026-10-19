"""Build a standalone acceptance figure from persisted metrics, without running inference."""

import hashlib
import json
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = Path("artifacts/stability-20261014")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


baseline = rows(root / "endurance-memory-baseline/samples.jsonl")
fixed = rows(root / "endurance/samples.jsonl")
run = json.loads((root / "endurance/run.json").read_text())
result_path = root / "endurance/report.json"
result = (
    json.loads(result_path.read_text())
    if result_path.exists()
    else {"passed": False, "state": "running"}
)
with sqlite3.connect("file:data/runs.sqlite3?mode=ro", uri=True) as db:
    events = [
        json.loads(r[0])
        for r in db.execute("SELECT body FROM events WHERE run_id=? ORDER BY seq", (run["id"],))
    ]
    first = db.execute(
        "SELECT body FROM predictions WHERE run_id=? ORDER BY rowid LIMIT 1", (run["id"],)
    ).fetchone()
    last = db.execute(
        "SELECT body FROM predictions WHERE run_id=? ORDER BY rowid DESC LIMIT 1", (run["id"],)
    ).fetchone()
latencies = [e for e in events if e["type"] == "prediction"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
colors = {"before": "#d97706", "after": "#2563eb"}
for data, label, color in [
    (baseline, "Before allocator bounds (stopped)", colors["before"]),
    (fixed, "Bounded allocator", colors["after"]),
]:
    if data:
        minutes = [r["elapsed_s"] / 60 for r in data]
        axes[0, 0].plot(
            minutes, [r["rss_mb"] for r in data], label=label, color=color, linewidth=1.2
        )
        base = next((r for r in data if r["elapsed_s"] >= 60), data[0])
        axes[1, 1].plot(
            minutes, [r["fds"] - base["fds"] for r in data], label=label, color=color, linewidth=1.2
        )
axes[0, 0].set(title="Web + collector resident memory", ylabel="RSS (MiB)")
axes[0, 0].legend(fontsize=8)
ages = []
if fixed:
    ages = [max((a for a in r["ages"].values() if a is not None), default=0) for r in fixed]
    axes[0, 1].plot(
        [r["elapsed_s"] / 60 for r in fixed], ages, color=colors["after"], linewidth=0.8
    )
axes[0, 1].axhline(0.75, color="#dc2626", linestyle="--", label="Input age gate: 0.75 s")
axes[0, 1].set(
    title="Oldest received input sample", ylabel="Age (s)", ylim=(0, max(1, max(ages, default=0)))
)
axes[0, 1].legend(fontsize=8)
axes[1, 0].plot(
    [(e["time_unix"] - run["created_unix"]) / 60 for e in latencies],
    [e["payload"]["rpc_s"] for e in latencies],
    color=colors["after"],
    linewidth=0.8,
)
axes[1, 0].axhline(3, color="#dc2626", linestyle="--", label="RPC P95 gate: 3 s")
axes[1, 0].set(title="RPC elapsed time (shared GPU service)", ylabel="Time (s)", ylim=(0, 3.3))
axes[1, 0].legend(fontsize=8)
axes[1, 1].axhline(10, color="#dc2626", linestyle="--", label="FD growth gate: 10")
axes[1, 1].set(title="File descriptors relative to 60 s baseline", ylabel="FD growth")
for ax in axes.flat:
    ax.set_xlabel("Elapsed time (minutes)")
    ax.grid(alpha=0.2)
fig.suptitle(
    "ForceVLA prediction inference acceptance: "
    + ("PASS" if result.get("passed") else "RUNNING / NOT YET PASSED"),
    fontsize=14,
)
fig.savefig(root / "acceptance.png", dpi=150)
plt.close(fig)
video = {}
if first and last:
    a, b = json.loads(first[0]), json.loads(last[0])
    one = a["input_metadata"]["decoder_stats"]
    two = b["input_metadata"]["decoder_stats"]
    video = {
        key: two[key] - one[key] for key in one if isinstance(one[key], (int, float)) and key in two
    }
    video.update(
        first_source_time=a["input_metadata"]["sources"]["image"]["source_time"],
        last_source_time=b["input_metadata"]["sources"]["image"]["source_time"],
    )
unit = Path("deploy/forcevla-web.service")
summary = {
    "run_id": run["id"],
    "endurance": result,
    "video_decoder_delta_between_first_and_last_prediction": video,
    "deployment_unit_sha256": hashlib.sha256(unit.read_bytes()).hexdigest(),
    "criteria_unchanged_from_baseline": json.loads((root / "endurance/criteria.json").read_text())
    == json.loads((root / "endurance-memory-baseline/criteria.json").read_text()),
}
(root / "acceptance-summary.json").write_text(
    json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "figure": str(root / "acceptance.png"),
            "criteria_unchanged": summary["criteria_unchanged_from_baseline"],
            "video_delta": video,
        },
        indent=2,
    )
)
