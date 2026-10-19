"""Constrained finite prefix on the original model time grid; no hardware IO.

The installed velocity-QP proposes references. A second, explicit constrained
least-squares problem enforces the V1 acceleration/start/stop contract. Every
result, including each possible interruption's braking path, is independently
checked in the robot geometry. Infeasible tracking is rejected, never slow-played.
"""

import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize
from scipy.spatial.transform import Rotation

from robot_control.action_adapter import prepare_actions
from robot_control.closed_loop_contracts import Segment
from robot_control.command_bridge import ControlFault
from robot_control.common import save_json


def brake_path(q, velocity, dt, acceleration):
    q, v = np.array(q, copy=True), np.array(velocity, copy=True)
    rows = []
    while np.max(np.abs(v)) > 1e-12:
        v = np.sign(v) * np.maximum(0, np.abs(v) - acceleration * dt)
        q = q + v * dt
        rows.append(q.copy())
    if not rows:
        rows.append(q.copy())
    return np.array(rows)


def constrained_prefix(
    reference, seed, initial_velocity, bounds, config, *, deadline=None
):
    """Optimize positions through velocity variables with linear hard constraints."""
    ref, seed, v0 = map(
        lambda a: np.asarray(a, dtype=float), (reference, seed, initial_velocity)
    )
    n = len(ref)
    if (
        n < 2
        or ref.shape != (n, 14)
        or seed.shape != (14,)
        or v0.shape != (14,)
        or not all(np.isfinite(x).all() for x in (ref, seed, v0))
    ):
        raise ControlFault("invalid_prefix_inputs")
    dt, speed, accel = (
        config.control_period_s,
        config.max_joint_speed_rad_s,
        config.max_joint_acceleration_rad_s2,
    )
    scale = config.max_joint_displacement_rad
    integrate = np.tril(np.ones((n, n))) * dt * speed / scale
    difference = np.eye(n) - np.eye(n, k=-1)
    limit = accel * dt / speed
    endpoint = np.eye(n)[-1:]
    hessian = 2 * (integrate.T @ integrate + 1e-7 * np.eye(n))
    joints = np.empty_like(ref)

    def timed(_=None):
        if deadline is not None and time.monotonic() > deadline:
            raise ControlFault("prepare_timeout")

    for j in range(14):
        timed()
        desired = (ref[:, j] - seed[j]) / scale
        linear = -2 * integrate.T @ desired
        low, high = np.full(n, -limit), np.full(n, limit)
        low[0] += v0[j] / speed
        high[0] += v0[j] / speed
        qmin = max(-1.0, (bounds[j, 0] - seed[j]) / scale)
        qmax = min(1.0, (bounds[j, 1] - seed[j]) / scale)
        result = minimize(
            lambda x: 0.5 * x @ hessian @ x + linear @ x,
            np.zeros(n),
            jac=lambda x: hessian @ x + linear,
            method="SLSQP",
            bounds=Bounds(-np.ones(n), np.ones(n)),
            constraints=[
                LinearConstraint(difference, low, high),
                LinearConstraint(integrate, qmin, qmax),
                LinearConstraint(endpoint, 0, 0),
            ],
            callback=timed,
            options={"ftol": 1e-10, "maxiter": 150},
        )
        if not result.success or not np.isfinite(result.x).all():
            raise ControlFault("constrained_prefix_infeasible:" + str(result.message))
        joints[:, j] = seed[j] + np.cumsum(result.x) * speed * dt
    return joints


def kinematic_checks(q, seed, velocity0, config, bounds, *, terminal=True):
    dt = config.control_period_s
    velocity = np.diff(np.vstack((seed, q)), axis=0) / dt
    velocities = (
        np.vstack((velocity0, velocity, np.zeros((1, 14))))
        if terminal
        else np.vstack((velocity0, velocity))
    )
    acceleration = np.diff(velocities, axis=0) / dt
    measures = dict(
        max_speed_rad_s=float(np.max(np.abs(velocity))),
        max_acceleration_rad_s2=float(np.max(np.abs(acceleration))),
        max_displacement_rad=float(np.max(np.abs(q - seed))),
    )
    checks = [
        (
            np.any(q < bounds[:, 0] - 1e-9) or np.any(q > bounds[:, 1] + 1e-9),
            "joint_limit",
        ),
        (measures["max_speed_rad_s"] > config.max_joint_speed_rad_s + 1e-7, "speed"),
        (
            measures["max_acceleration_rad_s2"]
            > config.max_joint_acceleration_rad_s2 + 1e-6,
            "acceleration",
        ),
        (
            measures["max_displacement_rad"] > config.max_joint_displacement_rad + 1e-9,
            "displacement",
        ),
    ]
    return [name for fail, name in checks if fail], measures, velocity


class PrefixPreparer:
    def __init__(self, solver, config):
        self.solver, self.config = solver, config

    def prepare(self, observation, prediction, identity, folder):
        c, solver = self.config, self.solver
        started = time.monotonic()
        deadline = min(
            started + c.prepare_timeout_s,
            observation.observed_at + c.max_prediction_age_s,
        )
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=False)
        raw = np.asarray(prediction["actions"])
        if prediction.get("request_id") != identity.request_id:
            raise ControlFault("prediction_request_mismatch")
        np.savez_compressed(folder / "raw_prediction.npz", actions=raw)
        bounded, adaptation = prepare_actions(raw, c.max_gripper_correction)
        save_json(folder / "action_adaptation.json", adaptation)
        if bounded is None:
            raise ControlFault("gripper_correction_excessive")
        if time.monotonic() >= deadline:
            raise ControlFault("prediction_expired_before_prepare")
        if np.max(np.abs(observation.velocities)) > c.stop_speed_rad_s:
            raise ControlFault("prepare_requires_stationary_feedback")
        qp = solver.rollout(
            bounded[0],
            observation.joints,
            observation.state,
            folder,
            prefix_steps=c.execution_horizon,
        )
        save_json(folder / "reference_qp_checks.json", qp)
        if qp.get("status") != "passed_scoped_qp" or qp.get("reasons"):
            raise ControlFault(
                "reference_qp_rejected:" + ",".join(qp.get("reasons", []))
            )
        with np.load(folder / "qp_trajectory.npz", allow_pickle=False) as data:
            reference = data["joint_positions"].copy()
            ids = data["target_indices"].copy()
            dt = float(data["dt"])
            original_seed = data["seed"].copy()
        n = math.ceil(
            c.execution_horizon * c.action_period_s / c.control_period_s - 1e-9
        )
        expected = np.floor(
            np.arange(n) * c.control_period_s / c.action_period_s + 1e-8
        ).astype(int)
        if (
            dt != c.control_period_s
            or reference.shape != (n, 14)
            or not np.array_equal(ids, expected)
            or not np.array_equal(original_seed, observation.joints)
        ):
            raise ControlFault("reference_qp_contract_mismatch")
        command_seed = np.array(
            observation.metadata.get("execution_start_target", observation.joints),
            dtype=float,
            copy=True,
        )
        if (
            command_seed.shape != (14,)
            or not np.isfinite(command_seed).all()
            or np.max(np.abs(command_seed - observation.joints)) > c.seed_error_rad
        ):
            raise ControlFault("hold_target_measured_seed_mismatch")
        # A normal hold sends a constant command. Preserve that command boundary
        # while the model and reference IK still use the new measured observation.
        command_velocity = np.zeros(14)
        q = constrained_prefix(
            reference,
            command_seed,
            command_velocity,
            solver.bounds,
            c,
            deadline=deadline,
        )
        reasons, measures, velocity = kinematic_checks(
            q, command_seed, command_velocity, c, solver.bounds
        )
        seed_poses = solver.poses(observation.joints)
        seed_rotations = Rotation.from_euler("xyz", seed_poses[:, 3:])
        min_distance, max_error_m, max_error_rad = float("inf"), 0.0, 0.0
        max_tcp_delta, max_tcp_rotation = 0.0, 0.0

        def check_geometry(row, target=None):
            nonlocal \
                min_distance, \
                max_error_m, \
                max_error_rad, \
                max_tcp_delta, \
                max_tcp_rotation
            if time.monotonic() >= deadline:
                raise ControlFault("prepare_timeout")
            if np.any(row < solver.bounds[:, 0]) or np.any(row > solver.bounds[:, 1]):
                reasons.append("braking_joint_limit")
            if (
                np.max(np.abs(row - observation.joints))
                > c.max_joint_displacement_rad + 1e-9
            ):
                reasons.append("braking_displacement")
            poses = solver.poses(row)
            rotations = Rotation.from_euler("xyz", poses[:, 3:])
            max_tcp_delta = max(
                max_tcp_delta,
                float(np.linalg.norm(poses[:, :3] - seed_poses[:, :3], axis=1).max()),
            )
            max_tcp_rotation = max(
                max_tcp_rotation,
                float((seed_rotations.inv() * rotations).magnitude().max()),
            )
            near = min(x["distance_m"] for x in solver.collisions(row))
            min_distance = min(min_distance, near)
            if target is not None:
                desired = np.stack((target[:6], target[7:13]))
                max_error_m = max(
                    max_error_m,
                    float(np.linalg.norm(poses[:, :3] - desired[:, :3], axis=1).max()),
                )
                max_error_rad = max(
                    max_error_rad,
                    float(
                        (Rotation.from_euler("xyz", desired[:, 3:]).inv() * rotations)
                        .magnitude()
                        .max()
                    ),
                )

        brakes = []
        for i, row in enumerate(q):
            check_geometry(row, bounded[0, ids[i]])
            brake = brake_path(row, velocity[i], dt, c.max_joint_acceleration_rad_s2)
            brake_reasons, _, _ = kinematic_checks(
                brake, row, velocity[i], c, solver.bounds
            )
            reasons.extend("brake_" + r for r in brake_reasons)
            for stop_row in brake:
                check_geometry(stop_row)
            brake.flags.writeable = False
            brakes.append(brake)
        if min_distance < solver.parameters["collision_minimum_distance"]:
            reasons.append("configured_collision_margin")
        if max_tcp_delta > c.max_tcp_displacement_m:
            reasons.append("tcp_displacement")
        if max_tcp_rotation > c.max_tcp_rotation_rad:
            reasons.append("tcp_rotation")
        if max_error_m > c.max_tcp_tracking_m or max_error_rad > c.max_tcp_tracking_rad:
            reasons.append("constrained_tcp_tracking_infeasible")
        grips = bounded[0, ids][:, [6, 13]].copy()
        grip_targets = bounded[0, : c.execution_horizon][:, [6, 13]]
        delta = np.diff(np.vstack((observation.gripper_targets, grip_targets)), axis=0)
        if np.max(np.abs(delta)) / c.action_period_s > c.max_gripper_speed_per_s:
            reasons.append("gripper_speed")
        if (
            np.max(np.abs(grips - observation.gripper_targets))
            > c.max_gripper_displacement
        ):
            reasons.append("gripper_displacement")
        if c.gripper_policy == "fixed_diagnostic" and not np.array_equal(
            grips, np.tile(observation.gripper_targets, (n, 1))
        ):
            reasons.append("fixed_diagnostic_requires_unchanged_prediction")
        max_brake_s = max(map(len, brakes)) * dt
        finish_age = time.monotonic() + n * dt + max_brake_s - observation.observed_at
        if finish_age > c.max_prediction_age_s:
            reasons.append("insufficient_execution_age_budget")
        report = dict(
            status="rejected" if reasons else "prepared",
            reasons=sorted(set(reasons)),
            **measures,
            max_tcp_displacement_m=max_tcp_delta,
            max_tcp_rotation_rad=max_tcp_rotation,
            max_tracking_m=max_error_m,
            max_tracking_rad=max_error_rad,
            min_configured_collision_distance_m=min_distance,
            max_brake_s=max_brake_s,
            prefix_steps=c.execution_horizon,
            ticks=n,
            dt=dt,
            prepare_s=time.monotonic() - started,
            predicted_finish_age_s=finish_age,
            config_sha256=c.digest(),
            identity=asdict(identity),
            time_policy="original_30Hz_no_retiming",
            environment_collision_checked=False,
            hardware_execution_verified=False,
            solver_manifest=solver.manifest,
            command_seed=command_seed.tolist(),
            measured_seed=observation.joints.tolist(),
            seed_policy="confirmed_command_target_with_measured_state_check",
        )
        np.savez_compressed(
            folder / "constrained_trajectory.npz",
            joints=q,
            grippers=grips,
            target_indices=ids,
            seed=command_seed,
            measured_seed=observation.joints,
            seed_velocity=command_velocity,
            dt=dt,
            **{f"brake_{i}": b for i, b in enumerate(brakes)},
        )
        save_json(folder / "checks.json", report)
        if reasons:
            raise ControlFault("prefix_rejected:" + ",".join(sorted(set(reasons))))
        for array in (q, grips, ids, command_seed, command_velocity):
            array.flags.writeable = False
        segment = Segment(
            identity,
            observation.observed_at,
            dt,
            command_seed,
            command_velocity,
            observation.gripper_targets,
            q,
            grips,
            ids,
            tuple(brakes),
            report,
        )
        save_json(folder / "plan.json", dict(report, plan_id=segment.digest()))
        return segment
