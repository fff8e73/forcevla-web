"""Fresh observation -> installed native QP -> immutable bounded joint prefix.

Preparation performs no device IO. Never relabel old observations or load a saved
plan as fresh. Production Stage B rejects moving targets at the transport layer.
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from robot_control.action_adapter import prepare_actions
from robot_control.command_bridge import ControlFault
from robot_control.common import save_json, sha256
from robot_control.joint_source_guard import JointEnvelope


@dataclass(frozen=True)
class PreparedJointPlan:
    request_id: str
    observed_at: float
    seed: np.ndarray
    joints: np.ndarray
    grippers: np.ndarray
    dt: float
    manifest: dict

    def envelope(self, token, index, sequence, command_time):
        # The scheduler must validate its deadline; no "freshening" on consumption.
        if not 0 <= index < len(self.joints):
            raise ControlFault("joint_plan_index")
        return JointEnvelope(
            token, sequence, command_time, self.observed_at, self.joints[index].copy()
        )


def prepare_native_plan(
    solver,
    *,
    actions,
    seed,
    state,
    request_id,
    observed_at,
    output,
    prefix_steps,
    max_correction,
    max_age_s=2.5,
    clock=time.monotonic,
):
    if (
        not request_id
        or not math.isfinite(observed_at)
        or not 0 <= clock() - observed_at <= max_age_s
    ):
        raise ControlFault("original_observation_stale")
    if type(prefix_steps) is not int or not 1 <= prefix_steps <= 50:
        raise ValueError("Invalid prefix length")
    folder = Path(output)
    folder.mkdir(parents=True, exist_ok=False)
    adapted, adaptation = prepare_actions(np.asarray(actions), max_correction)
    save_json(folder / "action_adaptation.json", adaptation)
    np.savez_compressed(folder / "raw_prediction.npz", actions=actions)
    if adapted is None:
        raise ControlFault("action_adaptation_rejected")
    result = solver.rollout(adapted[0], seed, state, folder)
    save_json(folder / "ik_checks.json", result)
    if (
        result.get("status") != "passed_scoped_qp"
        or result.get("reasons")
        or result.get("manifest", {}).get("same_installed_qp_core") is not True
    ):
        raise ControlFault("native_qp_checks_failed")
    if not 0 <= clock() - observed_at <= max_age_s:
        raise ControlFault("observation_expired_during_qp")
    with np.load(folder / "qp_trajectory.npz", allow_pickle=False) as data:
        q = data["joint_positions"].copy()
        ids = data["target_indices"].copy()
        original_seed = data["seed"].copy()
        names = data["joint_names"].tolist()
        dt = float(data["dt"])
    expected_names = [f"Joint{i}_{s}" for s in ("L", "R") for i in range(1, 8)]
    if (
        names != expected_names
        or not np.array_equal(original_seed, seed)
        or not np.isfinite(dt)
        or not 0 < dt <= 1 / 30
        or q.shape != (math.ceil(50 / 30 / dt - 1e-9), 14)
        or not np.isfinite(q).all()
    ):
        raise ControlFault("invalid_native_trajectory")
    expected = np.minimum(np.floor(np.arange(len(q)) * dt * 30 + 1e-8).astype(int), 49)
    if not np.array_equal(ids, expected):
        raise ControlFault("native_trajectory_timing_mismatch")
    keep = ids < prefix_steps
    joints = q[keep].copy()
    grippers = adapted[0, ids[keep]][:, [6, 13]].copy()
    original_seed.flags.writeable = joints.flags.writeable = (
        grippers.flags.writeable
    ) = False
    manifest = {
        "request_id": request_id,
        "original_observed_at": observed_at,
        "prefix_steps": prefix_steps,
        "ticks": len(joints),
        "dt": dt,
        "trajectory_sha256": sha256(folder / "qp_trajectory.npz"),
        "raw_prediction_sha256": sha256(folder / "raw_prediction.npz"),
        "solver_manifest": result["manifest"],
        "hardware_execution_ready": False,
        "scope": "QP-prepared prefix; prefix stopping/acceleration/environment require D validation",
    }
    save_json(folder / "prepared_plan.json", manifest)
    return PreparedJointPlan(
        request_id, observed_at, original_seed, joints, grippers, dt, manifest
    )
