"""Standalone installed QP-core rollout with an ABI-compatible private DAQP build.

No ROS initialization or hardware output. Online wrapper, mux and plant are not
simulated, and original deployment DAQP source revision has not been recovered.
"""

from pathlib import Path
import subprocess
import time

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from robot_control.common import CONTROL_ROOT, save_json, sha256
from robot_control.offline_ik import OfflineIK, JOINTS

BINARY = CONTROL_ROOT / "build/offline_qp"
CORE = Path(
    "/opt/kernelmind/apex/install/marvin_qp_controller/lib/marvin_qp_controller/libqp_controller_core.a"
)
DAQP_COMMIT = "653e22316c1d54ee7c71500f45f3ff7afaadad07"


class NativeQP(OfflineIK):
    def __init__(self, parameters, limits):
        super().__init__(parameters, limits)
        if not BINARY.is_file():
            raise RuntimeError(
                "Build the isolated native QP with build_native.sh first"
            )
        paths = [
            BINARY,
            CORE,
            CONTROL_ROOT / "build/libcompatible_daqp.a",
            CONTROL_ROOT / "native/offline_qp.cpp",
            CONTROL_ROOT / "native/CMakeLists.txt",
        ]
        paths += list((CONTROL_ROOT / "vendor/daqp/include").glob("*.h")) + list(
            (CONTROL_ROOT / "vendor/daqp/src").glob("*.c")
        )
        for path in paths:
            self.manifest["file_sha256"][str(path)] = sha256(path)
        self.manifest.update(
            solver="installed QpControllerCore + private DAQP v0.7.2 static build",
            same_installed_qp_core=True,
            daqp_source_commit=DAQP_COMMIT,
            daqp_settings_size=112,
            online_qp_equivalence_verified=False,
            scope="Installed QP static core and runtime options, same full model/constraints; private compatible DAQP. Full online wrapper/smoothing/mux/plant and original DAQP revision are not verified. Configured robot collision pairs only; environment unmodeled.",
        )

    def rollout(self, actions, seed, observed_state, output, *, prefix_steps=50):
        output = Path(output)
        actions = np.asarray(actions, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        state = np.asarray(observed_state, dtype=np.float64)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid 50x14 actions")
        if state.shape != (26,) or not np.isfinite(state).all():
            raise ValueError("Invalid observed state")
        if type(prefix_steps) is not int or not 1 <= prefix_steps <= 50:
            raise ValueError("Invalid prefix_steps")
        actions = actions[:prefix_steps]
        measured = self.poses(seed)
        observed = np.stack([state[:6], state[13:19]])
        seed_p = np.linalg.norm(measured[:, :3] - observed[:, :3], axis=1)
        seed_r = (
            Rotation.from_euler("xyz", observed[:, 3:]).inv()
            * Rotation.from_euler("xyz", measured[:, 3:])
        ).magnitude()
        reasons = []
        if np.any(seed < self.bounds[:, 0]) or np.any(seed > self.bounds[:, 1]):
            reasons.append("seed_joint_limit")
        if np.any(seed_p > self.limits["seed_position_tolerance_m"]) or np.any(
            seed_r > self.limits["seed_rotation_tolerance_rad"]
        ):
            reasons.append("seed_fk_observation_mismatch")
        initial = self.collisions(seed)
        if (
            min(d["distance_m"] for d in initial)
            < self.parameters["collision_minimum_distance"]
        ):
            reasons.append("seed_collision_margin")
        result = {
            "status": "rejected",
            "reasons": reasons,
            "manifest": self.manifest,
            "seed_errors": [
                {"position_m": float(p), "rotation_rad": float(r)}
                for p, r in zip(seed_p, seed_r)
            ],
            "seed_collision_distances": initial,
            "environment_collision_checked": False,
            "hardware_execution_ready": False,
            "reachability_interpretation": "Finite-time QP tracking, not global IK reachability. Includes intermediate controller ticks, excludes online wrapper/plant.",
        }
        if reasons:
            return result
        self.set_q(seed)
        base_r = self.data.xmat[self.base].reshape(3, 3).copy()
        base_p = self.data.xpos[self.base].copy()
        targets = []
        for action in actions:
            row = []
            for offset in [0, 7]:
                pose = action[offset : offset + 6]
                row.append(
                    np.r_[
                        base_p + base_r @ pose[:3],
                        Rotation.from_matrix(
                            base_r @ Rotation.from_euler("xyz", pose[3:]).as_matrix()
                        ).as_quat(),
                    ].tolist()
                )
            targets.append(row)
        request = {
            "parameters": self.parameters,
            "package_path": str(
                Path(
                    "/opt/kernelmind/apex/install/marvin_qp_controller/share/marvin_qp_controller"
                )
            ),
            "seed": dict(zip(JOINTS, seed.tolist())),
            "targets": targets,
            "action_period_s": 1 / 30,
        }
        save_json(output / "qp_input.json", request)
        start = time.perf_counter()
        try:
            run = subprocess.run(
                [
                    str(BINARY),
                    str(output / "qp_input.json"),
                    str(output / "qp_output.yaml"),
                ],
                capture_output=True,
                text=True,
                timeout=self.limits["solver_timeout_s"],
            )
            (output / "qp_stdout.log").write_text(run.stdout)
            (output / "qp_stderr.log").write_text(run.stderr)
        except subprocess.TimeoutExpired:
            result.update(
                reasons=["solver_timeout"],
                solve_wall_ms=(time.perf_counter() - start) * 1000,
            )
            return result
        result["native_process_ms"] = (time.perf_counter() - start) * 1000
        result["solver_returncode"] = run.returncode
        if not (output / "qp_output.yaml").is_file():
            result["reasons"].append("solver_process_failed")
            return result
        parse_start = time.perf_counter()
        loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
        native = yaml.load((output / "qp_output.yaml").read_text(), Loader=loader)
        result["trajectory_parse_ms"] = (time.perf_counter() - parse_start) * 1000
        result["trajectory_parser"] = loader.__name__
        expected_dt = 1 / self.parameters["control_frequency_hz"]
        if not np.isfinite(native["dt"]) or abs(native["dt"] - expected_dt) > 1e-12:
            raise RuntimeError("Unexpected native step period")
        if len(set(native["joint_names"])) != 14 or set(native["joint_names"]) != set(
            JOINTS
        ):
            raise RuntimeError("Unexpected native joint map")
        indices = native["q_indices"]
        if sorted(indices) != list(range(14)):
            raise RuntimeError("Invalid q indices")
        reorder = [indices[native["joint_names"].index(n)] for n in JOINTS]
        steps = native.get("steps", [])
        if not steps:
            result["reasons"].append("empty_trajectory")
            return result
        q = np.asarray([r["q"] for r in steps], dtype=float)
        if q.shape != (len(steps), 14) or not np.isfinite(q).all():
            raise RuntimeError("Invalid native trajectory")
        q = q[:, reorder]
        target_indices = np.array([s["target_index"] for s in steps], int)
        if (
            any(s["tick"] != i for i, s in enumerate(steps))
            or np.any(target_indices < 0)
            or np.any(target_indices >= prefix_steps)
        ):
            raise RuntimeError("Invalid tick/target identity")
        expected_indices = np.minimum(
            np.floor(np.arange(len(steps)) * expected_dt * 30 + 1e-8).astype(int), prefix_steps - 1
        )
        if not np.array_equal(target_indices, expected_indices):
            raise RuntimeError("Native target timing mismatch")
        velocities = np.abs(np.diff(np.vstack((seed, q)), axis=0)) / expected_dt
        violations = {
            "joint_limits": [],
            "velocity_limits": [],
            "configured_collision_margin": [],
            "joint_tick_discontinuity": [],
        }
        min_distance = 10.0
        closest_pair = None
        for tick, row in enumerate(q):
            if np.any(row < self.bounds[:, 0] - 1e-7) or np.any(
                row > self.bounds[:, 1] + 1e-7
            ):
                violations["joint_limits"].append(tick)
            if np.any(velocities[tick] > self.velocity + 1e-5):
                violations["velocity_limits"].append(tick)
            if (
                np.max(velocities[tick] * expected_dt)
                > self.limits["max_joint_step_rad"]
            ):
                violations["joint_tick_discontinuity"].append(tick)
            near = min(self.collisions(row), key=lambda d: d["distance_m"])
            if near["distance_m"] < min_distance:
                min_distance = near["distance_m"]
                closest_pair = near["pair"]
            if (
                near["distance_m"]
                < self.parameters["collision_minimum_distance"] - 1e-6
            ):
                violations["configured_collision_margin"].append(tick)
        records = []
        previous = seed
        for i in range(prefix_steps):
            ids = np.flatnonzero(target_indices == i)
            if not len(ids):
                break
            row = q[ids[-1]]
            actual = self.poses(row)
            target = np.stack([actions[i, :6], actions[i, 7:13]])
            ep = np.linalg.norm(actual[:, :3] - target[:, :3], axis=1)
            er = (
                Rotation.from_euler("xyz", target[:, 3:]).inv()
                * Rotation.from_euler("xyz", actual[:, 3:])
            ).magnitude()
            why = []
            if np.any(ep > self.limits["tracking_position_tolerance_m"]) or np.any(
                er > self.limits["tracking_rotation_tolerance_rad"]
            ):
                why.append("finite_time_tracking_error")
            jump = float(np.max(np.abs(row - previous)))
            previous = row
            if jump > self.limits["max_joint_step_rad"]:
                why.append("joint_continuity")
            records.append(
                {
                    "target_index": i,
                    "last_tick": int(ids[-1]),
                    "position_error_m": ep.tolist(),
                    "rotation_error_rad": er.tolist(),
                    "max_joint_change_rad": jump,
                    "reasons": why,
                }
            )
            reasons.extend(why)
        for name, ticks in violations.items():
            if ticks:
                reasons.append(name)
        if (
            run.returncode
            or native.get("status") != "completed"
            or not all(s["ok"] for s in steps)
        ):
            reasons.append("qp_failure")
        if (
            len(steps) != int(np.ceil(prefix_steps / 30 / expected_dt - 1e-9))
            or len(records) != prefix_steps
        ):
            reasons.append("incomplete_horizon")
        np.savez_compressed(
            output / "qp_trajectory.npz",
            joint_positions=q,
            target_indices=target_indices,
            seed=seed,
            joint_names=np.array(JOINTS),
            dt=np.array(expected_dt),
        )
        result.update(
            status="passed_scoped_qp" if not reasons else "rejected",
            reasons=sorted(set(reasons)),
            targets=records,
            simulated_ticks=len(steps),
            prefix_steps=prefix_steps,
            qp_dt_s=expected_dt,
            min_configured_collision_distance_m=min_distance,
            closest_pair=closest_pair,
            violations=violations,
            solve_wall_ms=(time.perf_counter() - start) * 1000,
        )
        return result
