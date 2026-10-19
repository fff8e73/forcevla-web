"""Offline bounded numerical IK on the complete training MJCF, no ROS or SDK.

This is a geometric diagnostic, not the installed online QP implementation.
Only configured robot collision pairs are checked; fixtures are not modeled.
"""

import importlib.metadata
import json
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robot_control.common import sha256

JOINTS = [f"Joint{i}_{side}" for side in ["L", "R"] for i in range(1, 8)]
PACKAGE = Path(
    "/opt/kernelmind/apex/install/marvin_qp_controller/share/marvin_qp_controller"
)


def pose_delta(target, current):
    a, b = np.asarray(target), np.asarray(current)
    return float(np.linalg.norm(a[:3] - b[:3])), float(
        (
            Rotation.from_euler("xyz", b[3:]).inv() * Rotation.from_euler("xyz", a[3:])
        ).magnitude()
    )


class OfflineIK:
    def __init__(self, parameters, limits):
        self.parameters, self.limits = parameters, limits
        scene = Path(parameters["scene_file"])
        self.model = mujoco.MjModel.from_xml_path(str(scene))
        self.data = mujoco.MjData(self.model)
        self.ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, x) for x in JOINTS
        ]
        if min(self.ids) < 0 or self.model.nq != 14:
            raise ValueError("Expected the complete 14-joint m6_696 model")
        self.qindices = self.model.jnt_qposadr[self.ids]
        self.bounds = self.model.jnt_range[self.ids].copy()
        self.tools = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, x)
            for x in ["left_tool", "right_tool"]
        ]
        self.base = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
        if min(self.tools + [self.base]) < 0:
            raise ValueError("Missing base or tool frame")
        pair_path = PACKAGE / "config" / parameters["collision_pairs_file"]
        velocity_path = PACKAGE / "config" / parameters["velocity_limits_file"]
        self.velocity = np.array(
            [
                json.loads(velocity_path.read_text())["velocity_limits"][n]
                for n in JOINTS
            ]
        )
        self.pairs = []
        for left, right in json.loads(pair_path.read_text())["collision_pairs"]:
            for a in left:
                for b in right:
                    ids = [
                        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, n)
                        for n in [a, b]
                    ]
                    if min(ids) < 0:
                        raise ValueError("Missing collision geometry")
                    self.pairs.append((a, b, *ids))
        paths = [
            scene,
            pair_path,
            velocity_path,
            PACKAGE / "config" / parameters["ik_tasks_file"],
        ]
        paths.extend(
            (scene.parent / e.attrib["file"]).resolve()
            for e in ET.parse(scene).getroot().findall(".//mesh")
            if "file" in e.attrib
        )
        self.manifest = {
            "file_sha256": {str(p): sha256(p) for p in paths},
            "versions": {
                n: importlib.metadata.version(n) for n in ["mujoco", "scipy", "numpy"]
            },
            "solver": "scipy.optimize.least_squares bounded TRF, sequential measured seed, rotation-vector residual",
            "online_qp_equivalence_verified": False,
            "scope": "Complete same MJCF/tool frames/joint bounds; same configured collision pairs and velocity limits checked after IK. Not online QP, not dynamics, not continuous collision proof; fixtures, gripper geometry and environment unmodeled.",
        }

    def set_q(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (14,) or not np.isfinite(q).all():
            raise ValueError("Invalid measured seed")
        self.data.qpos[self.qindices] = q
        mujoco.mj_kinematics(self.model, self.data)

    def transforms(self, q):
        self.set_q(q)
        base_r = self.data.xmat[self.base].reshape(3, 3)
        base_p = self.data.xpos[self.base]
        return [
            (
                base_r.T @ (self.data.xpos[t] - base_p),
                base_r.T @ self.data.xmat[t].reshape(3, 3),
            )
            for t in self.tools
        ]

    def poses(self, q):
        return np.array(
            [
                np.r_[p, Rotation.from_matrix(r).as_euler("xyz")]
                for p, r in self.transforms(q)
            ]
        )

    def collisions(self, q):
        self.set_q(q)
        distances = []
        for a, b, ia, ib in self.pairs:
            distance = float(
                mujoco.mj_geomDistance(self.model, self.data, ia, ib, 10.0, None)
            )
            distances.append({"pair": [a, b], "distance_m": distance})
        return distances

    def rollout(self, actions, seed, observed_state, output):
        output = Path(output)
        actions = np.asarray(actions, dtype=np.float64)
        seed = np.asarray(seed, dtype=np.float64)
        state = np.asarray(observed_state, dtype=np.float64)
        if actions.shape != (50, 14) or not np.isfinite(actions).all():
            raise ValueError("Invalid 50x14 actions")
        if state.shape != (26,) or not np.isfinite(state).all():
            raise ValueError("Invalid observed state")
        observed = np.stack([state[:6], state[13:19]])
        seed_errors = [pose_delta(a, b) for a, b in zip(self.poses(seed), observed)]
        reasons = []
        if np.any(seed < self.bounds[:, 0]) or np.any(seed > self.bounds[:, 1]):
            reasons.append("seed_joint_limit")
        if any(
            p > self.limits["seed_position_tolerance_m"]
            or r > self.limits["seed_rotation_tolerance_rad"]
            for p, r in seed_errors
        ):
            reasons.append("seed_fk_observation_mismatch")
        distances = self.collisions(seed)
        closest = min(distances, key=lambda x: x["distance_m"])
        if closest["distance_m"] < self.parameters["collision_minimum_distance"]:
            reasons.append("seed_collision_margin")
        result = {
            "status": "rejected",
            "seed_errors": [
                {"position_m": p, "rotation_rad": r} for p, r in seed_errors
            ],
            "seed_collision_distances": distances,
            "reasons": reasons,
            "manifest": self.manifest,
            "environment_collision_checked": False,
            "hardware_execution_ready": False,
            "reachability_interpretation": "Local numerical IK failure is not proof of global unreachability. Sequential solution uses previous candidate, not measured future state.",
        }
        if reasons:
            return result
        start = time.perf_counter()
        deadline = start + self.limits["solver_timeout_s"]
        previous = seed.copy()
        rows = []
        trajectory = []
        path_checks = 0
        min_distance = closest["distance_m"]
        closest_pair = closest["pair"]
        for i, action in enumerate(actions):
            target = [action[:6], action[7:13]]
            target_r = [Rotation.from_euler("xyz", p[3:]) for p in target]

            def residual(q):
                if time.perf_counter() > deadline:
                    raise TimeoutError("Offline IK total time budget exceeded")
                transforms = self.transforms(q)
                err = []
                for (pos, rot), goal, rotation in zip(transforms, target, target_r):
                    err.extend(
                        (pos - goal[:3]) / self.limits["tracking_position_tolerance_m"]
                    )
                    err.extend(
                        (rotation.inv() * Rotation.from_matrix(rot)).as_rotvec()
                        / self.limits["tracking_rotation_tolerance_rad"]
                    )
                # Tie-break redundant configurations near the preceding solution.
                return np.r_[
                    err, self.limits["posture_regularization"] * (q - previous)
                ]

            try:
                fit = least_squares(
                    residual,
                    np.clip(
                        previous, self.bounds[:, 0] + 1e-9, self.bounds[:, 1] - 1e-9
                    ),
                    bounds=(self.bounds[:, 0], self.bounds[:, 1]),
                    max_nfev=self.limits["max_nfev"],
                    ftol=1e-6,
                    xtol=1e-6,
                    gtol=1e-6,
                )
            except TimeoutError:
                reasons.append("solver_timeout")
                break
            q = fit.x
            errors = [pose_delta(a, b) for a, b in zip(target, self.poses(q))]
            change = np.abs(q - previous)
            violations = []
            if not fit.success:
                violations.append("local_solver_not_converged")
            if any(
                p > self.limits["tracking_position_tolerance_m"]
                or r > self.limits["tracking_rotation_tolerance_rad"]
                for p, r in errors
            ):
                violations.append("ik_residual")
            if np.any(q < self.bounds[:, 0] - 1e-9) or np.any(
                q > self.bounds[:, 1] + 1e-9
            ):
                violations.append("joint_limits")
            if np.max(change) > self.limits["max_joint_step_rad"]:
                violations.append("joint_continuity")
            if np.any(change * 30 > self.velocity):
                violations.append("nominal_joint_velocity")
            # These are collision probes along a diagnostic joint segment, not
            # commands or a claim to reproduce the online controller trajectory.
            count = max(
                1,
                int(np.ceil(np.max(change) / self.limits["collision_probe_step_rad"])),
            )
            collision_violation = False
            for f in np.linspace(0, 1, count + 1)[1:]:
                distances = self.collisions(previous + (q - previous) * f)
                path_checks += 1
                near = min(distances, key=lambda x: x["distance_m"])
                if near["distance_m"] < min_distance:
                    min_distance = near["distance_m"]
                    closest_pair = near["pair"]
                if near["distance_m"] < self.parameters["collision_minimum_distance"]:
                    collision_violation = True
            if collision_violation:
                violations.append("configured_collision_margin")
            rows.append(
                {
                    "target_index": i,
                    "solver_success": bool(fit.success),
                    "nfev": fit.nfev,
                    "position_error_m": [p for p, r in errors],
                    "rotation_error_rad": [r for p, r in errors],
                    "max_joint_change_rad": float(np.max(change)),
                    "max_velocity_ratio": float(np.max(change * 30 / self.velocity)),
                    "reasons": violations,
                }
            )
            reasons.extend(violations)
            trajectory.append(q.copy())
            previous = q
        if len(rows) != 50:
            reasons.append("incomplete_horizon")
        np.savez_compressed(
            output / "ik_trajectory.npz",
            joint_positions=np.asarray(trajectory),
            seed=seed,
            joint_names=np.array(JOINTS),
        )
        result.update(
            status="passed_scoped_ik" if not reasons else "rejected",
            reasons=sorted(set(reasons)),
            targets=rows,
            min_configured_collision_distance_m=min_distance,
            closest_pair=closest_pair,
            collision_probes=path_checks,
            solve_wall_ms=(time.perf_counter() - start) * 1000,
            action_period_s=1 / 30,
        )
        return result
