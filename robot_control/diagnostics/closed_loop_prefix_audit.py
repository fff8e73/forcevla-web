"""Reproduce historical start/stop failures offline; never connect to hardware."""

import json
import sys
import time
from pathlib import Path

import numpy as np

from robot_control.closed_loop_contracts import LoopConfig
from robot_control.closed_loop_planning import constrained_prefix, kinematic_checks
from robot_control.common import save_json
from robot_control.native_qp import NativeQP

root = Path(__file__).resolve().parents[2]
source = root / "data/integration-runs/stage-cd-after-hold-prepare-01"
files = list(source.rglob("qp_trajectory.npz"))
if not files:
    raise SystemExit("Historical trajectory missing")
limits = json.loads((root / "config/planning.json").read_text())
solver = NativeQP(limits["expected_qp_parameters"], limits)
with np.load(files[0], allow_pickle=False) as d:
    q = d["joint_positions"][d["target_indices"] < 1].copy()
    seed = d["seed"].copy()
config = LoopConfig(execution_horizon=1)
raw_fail, raw_metrics, _ = kinematic_checks(
    q, seed, np.zeros(14), config, solver.bounds
)
started = time.monotonic()
constrained = constrained_prefix(q, seed, np.zeros(14), solver.bounds, config)
fail, metrics, _ = kinematic_checks(
    constrained, seed, np.zeros(14), config, solver.bounds
)
report = dict(
    source=str(files[0]),
    raw_metrics=raw_metrics,
    raw_failures=raw_fail,
    constrained_metrics=metrics,
    constrained_failures=fail,
    seconds=time.monotonic() - started,
    original_dt_s=0.002,
    retimed=False,
    hardware_commands_sent=0,
    diagnosis="Installed QP exposes velocity constraints but no acceleration or finite-prefix stop boundary. Directly truncating its response produces the recorded entry/exit derivatives. Constrained preparation enforces both boundaries; TCP tracking and collision must still pass independently.",
    execution_authorized=False,
)
path = root / "artifacts/robot-control/closed-loop-v1/prefix-audit.json"
save_json(path, report)
print(json.dumps(report, ensure_ascii=False, indent=2))
