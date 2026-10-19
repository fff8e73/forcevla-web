"""Explicitly establish a NEW fixed hold target from measured joints.

Not preservation/readback of the previous controller goal. Position mode only;
no prediction, no trajectory, no gripper target changes. Missing mux targets stay
NaN/time0 until actual mux output arrives; they cannot confirm a hold.
"""
import uuid

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import JointExecutionBackend


class MeasuredHoldBackend(JointExecutionBackend):
    measured_hold = True

    def __init__(self, *args, gripper_monitor_limit_rad_s=None, **kwargs):
        super().__init__(*args, **kwargs)
        # Only this explicitly requested fixed-target experiment can override the
        # outer monitor. The installed driver's claim threshold is unchanged.
        self.gripper_monitor_override = gripper_monitor_limit_rad_s is not None
        if gripper_monitor_limit_rad_s is not None:
            if (type(gripper_monitor_limit_rad_s) not in (float, int)
                    or not np.isfinite(gripper_monitor_limit_rad_s)
                    or not 0 < gripper_monitor_limit_rad_s <= .5):
                raise ValueError('Invalid measured-hold gripper monitor limit')
            self.gripper_stop_speed = float(gripper_monitor_limit_rad_s)
        self.gripper_initial_actual = None


    def checked_feedback(self, *, owned=False):
        f = super().checked_feedback(owned=owned)
        if f.mode != 1:
            raise ControlFault('measured_hold_requires_position_mode')
        if np.any(np.abs(f.velocities) > self.stop_speed):
            raise ControlFault('measured_hold_joint_speed')
        if self.last_q is not None and np.any(np.abs(f.joints-self.last_q) > self.tracking):
            raise ControlFault('measured_hold_joint_displacement')
        if self.grip_target is not None and not np.allclose(
                f.gripper['applied_target'], self.grip_target, atol=1e-8, rtol=0):
            raise ControlFault('measured_hold_gripper_target_changed')
        if self.gripper_initial_actual is not None:
            calibration = np.asarray(f.gripper['calibration'])
            displacement = np.abs(np.asarray(f.gripper['actual'])-self.gripper_initial_actual) * (calibration[:,1]-calibration[:,0])
            if np.any(displacement > .01):
                raise ControlFault('measured_hold_gripper_displacement')
            if np.any(np.abs(f.gripper['motor_velocities']) > self.gripper_stop_speed):
                raise ControlFault('measured_hold_gripper_monitor_speed')
        return f

    def acquire(self, owner):
        with self.lock:
            if self.state != 'IDLE':
                raise ControlFault('session_requires_review_or_already_used')
            if self.thread is None:
                raise ControlFault('measured_hold_watchdog_not_running')
            # Caller must check planner/replay/site readiness before invoking.
            f = self.checked_feedback()
            if f.selected_input not in (0,2) or f.gripper.get('reserved') is not False:
                raise ControlFault('measured_hold_requires_none_or_idle_planner')
            if np.any(np.abs(f.gripper['motor_velocities']) > self.gripper_stop_speed):
                raise ControlFault('gripper_not_stationary')
            self.initial_mode = 1
            self.state = 'ACQUIRING'
            self.record('measured_hold_intent', owner=owner, previous_input=f.selected_input,
                        policy='new_target_from_measured_joints_not_original_target',
                        gripper_monitor_limit_rad_s=self.gripper_stop_speed,
                        driver_claim_speed_limit_rad_s=.02,
                        joint_speed_limit_rad_s=self.stop_speed)
            try:
                self.io.select_input(0)
                f = self.checked_feedback()
                if f.selected_input != 0:
                    raise ControlFault('input_none_unconfirmed')
                # Snapshot once, after None readback. Never chase changing feedback.
                self.last_q = self.guard.validate_q(f.joints)
                self.seed_observed_at = f.received_at
                self.grip_target = np.array(f.gripper['applied_target'],copy=True)
                self.gripper_initial_actual = np.array(f.gripper['actual'],copy=True)
                self.boot, self.session = f.gripper['boot_id'], str(uuid.uuid4())
                self.record('fixed_measured_seed', joints=self.last_q.tolist(),
                            observed_at=self.seed_observed_at, grippers=self.grip_target.tolist())
                self.claim_attempted = True
                g = self.grip('claim')
                if not np.allclose(g['applied_target'],self.grip_target,atol=1e-9,rtol=0):
                    raise ControlFault('claim_changed_target')
                self.checked_feedback()  # verify displacement during claim
                self.io.bind_stationary_target(self.last_q,self.grip_target)
                self.selection_attempted = True
                self.record('select_user_intent')
                self.io.select_input(3)
                f = self.checked_feedback(owned=True)
                self.token = self.guard.claim(owner,self.clock(),self.last_q,f.received_at,3)
                self.state='ACTIVE'
                self.record('acquired_measured_hold')
                return self.token
            except Exception as exc:
                self.trip('measured_hold_acquire_failed:'+str(exc))
                raise
