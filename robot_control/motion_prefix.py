"""Stage D finite-prefix checks and deadline scheduling. No implicit live arming."""

import math
import json
import hashlib
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robot_control.command_bridge import ControlFault


def plan_digest(plan):
    h = hashlib.sha256(
        json.dumps(
            {
                "request_id": plan.request_id,
                "observed_at": plan.observed_at,
                "dt": plan.dt,
                "manifest": plan.manifest,
            },
            sort_keys=True,
            allow_nan=False,
        ).encode()
    )
    for value in [plan.seed, plan.joints, plan.grippers]:
        a = np.ascontiguousarray(value, dtype=np.float64)
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def profile_digest(profile):
    return hashlib.sha256(
        json.dumps(profile, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def validate_trial_profile(profile, *, diagnostic_only=False):
    for key in [
        "max_joint_displacement_rad",
        "max_joint_speed_rad_s",
        "max_joint_acceleration_rad_s2",
        "max_tcp_displacement_m",
        "max_tcp_rotation_rad",
        "max_gripper_displacement",
        "max_gripper_speed_per_s",
        "max_prediction_age_s",
        "max_lateness_s",
        "max_feedback_seed_error_rad",
        "max_initial_speed_rad_s",
    ]:
        value = profile.get(key)
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid trial budget: " + key)
    if (
        type(profile.get("prefix_steps")) is not int
        or not 1 <= profile["prefix_steps"] <= (50 if diagnostic_only else 3)
    ):
        raise ValueError("Execution prefix must be 1..3 steps; diagnostics allow 1..50")
    for key in [
        "site_trial_limits_accepted",
        "independent_stop_path_verified",
        "require_b_live_hold_acceptance",
    ]:
        if type(profile.get(key)) is not bool:
            raise ValueError("Explicit evidence flag required: " + key)
    if profile["require_b_live_hold_acceptance"] is not True:
        raise ValueError("B live acceptance cannot be disabled")


def check_motion_prefix(
    plan, solver, profile, *, seed_velocity, initial_gripper_targets, diagnostic_only=False
):
    validate_trial_profile(profile, diagnostic_only=diagnostic_only)
    q = np.asarray(plan.joints)
    seed = np.asarray(plan.seed)
    grip = np.asarray(plan.grippers)
    v0 = np.asarray(seed_velocity, dtype=float)
    g0 = np.asarray(initial_gripper_targets, dtype=float)
    reasons = []
    if (
        seed.shape != (14,)
        or not np.isfinite(seed).all()
        or not math.isfinite(plan.dt)
        or not 0 < plan.dt <= 1 / 30
        or len(q) == 0
    ):
        raise ControlFault("invalid_plan_seed_or_timing")
    if plan.manifest["prefix_steps"] != profile["prefix_steps"]:
        raise ControlFault("prefix_profile_mismatch")
    if (
        v0.shape != (14,)
        or not np.isfinite(v0).all()
        or g0.shape != (2,)
        or not np.isfinite(g0).all()
    ):
        raise ControlFault("invalid_trial_initial_feedback")
    if np.any(g0 < 0) or np.any(g0 > 1):
        raise ControlFault("invalid_initial_gripper_target")
    if (
        q.ndim != 2
        or q.shape[1] != 14
        or grip.shape != (len(q), 2)
        or not np.isfinite(q).all()
        or not np.isfinite(grip).all()
    ):
        raise ControlFault("invalid_prepared_plan")
    if np.any(np.abs(v0) > profile["max_initial_speed_rad_s"]):
        reasons.append("initial_motion_not_stationary")
    delta = q - seed
    velocity = np.diff(np.vstack((seed, q)), axis=0) / plan.dt
    acceleration = (
        np.diff(np.vstack((v0, velocity, np.zeros((1, 14)))), axis=0) / plan.dt
    )
    joint_displacement = float(np.max(np.abs(delta)))
    speed = float(np.max(np.abs(velocity)))
    accel = float(np.max(np.abs(acceleration)))
    if joint_displacement > profile["max_joint_displacement_rad"]:
        reasons.append("trial_joint_displacement")
    if speed > profile["max_joint_speed_rad_s"]:
        reasons.append("trial_joint_speed")
    if accel > profile["max_joint_acceleration_rad_s2"]:
        reasons.append("trial_acceleration_or_stop_boundary")
    seed_pose = solver.poses(seed)
    seed_rot = Rotation.from_euler("xyz", seed_pose[:, 3:])
    max_position = max_rotation = 0.0
    for row in q:
        pose = solver.poses(row)
        max_position = max(
            max_position,
            float(np.linalg.norm(pose[:, :3] - seed_pose[:, :3], axis=1).max()),
        )
        max_rotation = max(
            max_rotation,
            float(
                (seed_rot.inv() * Rotation.from_euler("xyz", pose[:, 3:]))
                .magnitude()
                .max()
            ),
        )
    if max_position > profile["max_tcp_displacement_m"]:
        reasons.append("trial_tcp_displacement")
    if max_rotation > profile["max_tcp_rotation_rad"]:
        reasons.append("trial_tcp_rotation")
    # Target changes occur on the model's 30Hz grid, not arbitrary slow playback.
    first = np.r_[True, np.any(np.diff(grip, axis=0) != 0, axis=1)]
    indices = np.flatnonzero(first)
    grip_speed = 0.0
    previous = g0
    previous_time = -1 / 30
    for i in indices:
        at = i * plan.dt
        grip_speed = max(
            grip_speed, float(np.max(np.abs(grip[i] - previous))) / (at - previous_time)
        )
        previous = grip[i]
        previous_time = at
    grip_delta = float(np.max(np.abs(grip - g0)))
    if grip_delta > profile["max_gripper_displacement"]:
        reasons.append("trial_gripper_displacement")
    if grip_speed > profile["max_gripper_speed_per_s"]:
        reasons.append("trial_gripper_speed")
    return {
        "status": "passed_prefix_budgets" if not reasons else "rejected",
        "plan_sha256": plan_digest(plan),
        "profile_sha256": profile_digest(profile),
        "reasons": reasons,
        "max_joint_displacement_rad": joint_displacement,
        "max_joint_speed_rad_s": speed,
        "max_joint_acceleration_including_stop_rad_s2": accel,
        "max_tcp_displacement_m": max_position,
        "max_tcp_rotation_rad": max_rotation,
        "max_gripper_displacement": grip_delta,
        "max_gripper_target_speed_per_s": grip_speed,
        "ticks": len(q),
        "duration_s": len(q) * plan.dt,
        "terminal_target_hold_checked": True,
        "diagnostic_only": diagnostic_only,
        "hardware_execution_ready": False,
        "scope": "command trajectory budget, not physical braking proof",
    }


def hardware_trial_blockers(stage_b_report, profile):
    validate_trial_profile(profile)
    reasons = []
    # A real fixed-target B cycle is a prerequisite for the first motion trial,
    # not proof of moving stop or full fault-injection acceptance.
    b = stage_b_report or {}
    fixed_hold = (
        b.get("status") == "passed_measured_joint_fixed_hold"
        and b.get("live_requested") is True
        and b.get("final_state") == "RELEASED"
        and b.get("submitted_pairs", 0) > 0
        and b.get("production_publications", 0) > 0
        and b.get("stop_report", {}).get("hold_feedback_confirmed") is True
        and b.get("final_feedback", {}).get("mode") == 1
        and b.get("final_feedback", {}).get("input") == 0
    )
    if not fixed_hold and not (
        b.get("stage_b_hardware_acceptance_passed") is True and b.get("domain") == 24
    ):
        reasons.append("stage_B_live_hold_acceptance_missing")
    if not profile["site_trial_limits_accepted"]:
        reasons.append("site_trial_limits_not_accepted")
    if not profile["independent_stop_path_verified"]:
        reasons.append("independent_stop_path_not_verified")
    return reasons


class PrefixScheduler:
    """Caller-owned deadline loop; supports certified nonblocking adapters only.

    Stage B's synchronous/fsync-per-tick backend intentionally does not advertise
    these capabilities. Sandbox implementations exercise the contract; production
    remains stationary-only. No slow playback or timestamp freshening fallback.
    """

    def __init__(self, backend, profile, *, clock=time.monotonic, sleep=time.sleep):
        validate_trial_profile(profile)
        self.backend, self.profile, self.clock, self.sleep = (
            backend,
            profile,
            clock,
            sleep,
        )
        self.used = False
        self.events = []

    def execute(
        self, plan, token, prefix_check, *, stage_b_report=None, cancel=lambda: False
    ):
        if self.used:
            raise ControlFault("scheduler_single_use")
        self.used = True
        if prefix_check.get("diagnostic_only"):
            raise ControlFault("diagnostic_plan_cannot_execute")
        if prefix_check.get("status") != "passed_prefix_budgets" or prefix_check.get(
            "reasons"
        ):
            raise ControlFault("prefix_checks_failed")
        if prefix_check.get("plan_sha256") != plan_digest(plan) or prefix_check.get(
            "profile_sha256"
        ) != profile_digest(self.profile):
            raise ControlFault("plan_or_profile_changed_after_validation")
        if getattr(self.backend.io, "production", False):
            reasons = hardware_trial_blockers(stage_b_report, self.profile)
            if reasons:
                raise ControlFault(";".join(reasons))
        capabilities = getattr(self.backend, "scheduler_capabilities", {})
        if (
            capabilities.get("bounded_joint_submit") is not True
            or capabilities.get("independent_gripper_lane") is not True
        ):
            raise ControlFault(
                "stage_B_synchronous_transport_not_qualified_for_motion_scheduler"
            )
        now = self.clock()
        if (
            not 0 <= now - plan.observed_at
            or now - plan.observed_at + len(plan.joints) * plan.dt
            > self.profile["max_prediction_age_s"]
        ):
            raise ControlFault("prefix_would_expire_before_finish")
        feedback = self.backend.checked_feedback(owned=True)
        if (
            np.max(np.abs(feedback.joints - plan.seed))
            > self.profile["max_feedback_seed_error_rad"]
        ):
            raise ControlFault("current_seed_changed_since_observation")
        start = self.clock()
        last_heartbeat = start
        grip_last = None
        report = {
            "status": "failed",
            "sent_joint_ticks": 0,
            "hardware_stop_verified": False,
            "events": self.events,
        }
        try:
            for i in range(len(plan.joints)):
                due = start + (i + 1) * plan.dt
                delay = due - self.clock()
                if delay > 0:
                    self.sleep(delay)
                now = self.clock()
                if cancel():
                    raise ControlFault("operator_cancelled")
                if now - due > self.profile["max_lateness_s"]:
                    raise ControlFault("scheduler_deadline_missed")
                if now - plan.observed_at > self.profile["max_prediction_age_s"]:
                    raise ControlFault("prediction_expired")
                if now - last_heartbeat >= 0.05:
                    self.backend.heartbeat(token)
                    last_heartbeat = self.clock()
                if grip_last is None or not np.array_equal(grip_last, plan.grippers[i]):
                    self.backend.queue_gripper(
                        token, plan.grippers[i].copy(), plan.observed_at, due
                    )
                    grip_last = plan.grippers[i].copy()
                self.backend.write_joint_tick(plan.envelope(token, i, i, now))
                finished = self.clock()
                self.events.append(
                    {
                        "tick": i,
                        "deadline": due,
                        "submitted_at": now,
                        "finished_at": finished,
                        "original_observed_at": plan.observed_at,
                    }
                )
                report["sent_joint_ticks"] += 1
                if finished - due > self.profile["max_lateness_s"]:
                    raise ControlFault("submission_deadline_missed")
            if getattr(self.backend, "gripper_pending", False):
                raise ControlFault("gripper_ack_not_complete_before_prefix_end")
            self.backend.trip("finite_prefix_complete")
            report["status"] = "prefix_submitted_hold_requested"
        except Exception as exc:
            report["error"] = str(exc)
            self.backend.trip("prefix_fault:" + str(exc))
        return report
