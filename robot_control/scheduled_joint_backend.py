"""Stage-D bounded submission adapter; sandbox qualification only.

State/ownership transitions keep Stage B's durable journal. High-rate command
records use a bounded asynchronous JSONL writer. Gripper receipt waits never run
on the joint-submission thread. Production remains stationary-only.
"""

import json
import os
from pathlib import Path
import queue
import threading
import uuid

import numpy as np

from robot_control.command_bridge import ControlFault
from robot_control.joint_execution import JointExecutionBackend


class ScheduledJointBackend(JointExecutionBackend):
    scheduler_capabilities = {
        "bounded_joint_submit": True,
        "independent_gripper_lane": True,
        "production_motion_qualified": False,
    }

    def __init__(self, io, bounds, velocities, journal, **kwargs):
        if getattr(io, "production", False):
            raise ControlFault(
                "live_scheduled_motion_requires_B_acceptance_and_transport_qualification"
            )
        self.audit_queue = queue.Queue(maxsize=4096)
        self.gripper_queue = queue.Queue(maxsize=1)
        self.lane_stop = threading.Event()
        self.async_error = None
        self.gripper_pending = False
        super().__init__(io, bounds, velocities, journal, **kwargs)
        self.audit_path = Path(journal).with_suffix(".ticks.jsonl")
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        self.audit_thread = threading.Thread(
            target=self.audit_writer, name="joint-audit-writer", daemon=True
        )
        self.gripper_thread = threading.Thread(
            target=self.gripper_worker, name="gripper-receipts", daemon=True
        )
        self.audit_thread.start()
        self.gripper_thread.start()

    def record(self, event, **fields):
        if event in {
            "submit_intent",
            "transport_submission",
            "gripper_lane_request",
            "gripper_lane_ack",
        }:
            if self.async_error:
                raise ControlFault("async_lane_failure:" + self.async_error)
            try:
                self.audit_queue.put_nowait(
                    {
                        "time": self.clock(),
                        "event": event,
                        "state": self.state,
                        **fields,
                    }
                )
            except queue.Full:
                raise ControlFault("audit_queue_full") from None
        else:
            super().record(event, **fields)

    def audit_writer(self):
        try:
            with self.audit_path.open("x") as f:
                while not self.lane_stop.is_set() or not self.audit_queue.empty():
                    try:
                        row = self.audit_queue.get(timeout=0.02)
                    except queue.Empty:
                        continue
                    f.write(json.dumps(row, allow_nan=False) + "\n")
                    while True:
                        try:
                            row = self.audit_queue.get_nowait()
                            f.write(json.dumps(row, allow_nan=False) + "\n")
                        except queue.Empty:
                            break
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as exc:
            self.async_error = "audit:" + repr(exc)

    def write_joint_tick(self, envelope):
        # No changed gripper target is sent by this synchronous path.
        return super().write(envelope, grippers=None)

    def queue_gripper(self, token, values, observed_at, due):
        values = np.array(values, dtype=float, copy=True)
        if (
            values.shape != (2,)
            or not np.isfinite(values).all()
            or np.any(values < 0)
            or np.any(values > 1)
        ):
            raise ControlFault("invalid_gripper_target")
        with self.lock:
            self.check_token(token)
            if self.state != "ACTIVE":
                raise ControlFault("session_not_active")
            if np.array_equal(values, self.grip_target) and not self.gripper_pending:
                return
            if self.gripper_pending:
                raise ControlFault("gripper_lane_busy_no_stale_queue")
            if not 0 <= self.clock() - observed_at <= self.guard.prediction_max_age_s:
                raise ControlFault("gripper_prediction_expired")
            self.gripper_pending = True
            try:
                self.gripper_queue.put_nowait((token, values, observed_at, due))
            except queue.Full:
                self.gripper_pending = False
                raise ControlFault("gripper_lane_full") from None

    def gripper_worker(self):
        while not self.lane_stop.is_set():
            try:
                token, values, observed, due = self.gripper_queue.get(timeout=0.02)
            except queue.Empty:
                continue
            try:
                for index, side in enumerate(["left", "right"]):
                    with self.lock:
                        self.check_token(token)
                        if self.state != "ACTIVE":
                            raise ControlFault("gripper_lane_cancelled")
                        if (
                            not 0
                            <= self.clock() - observed
                            <= self.guard.prediction_max_age_s
                        ):
                            raise ControlFault("gripper_prediction_expired")
                        if values[index] == self.grip_target[index]:
                            continue
                        f = self.checked_feedback(owned=True)
                        self.grip_seq += 1
                        request = {
                            "action": "target",
                            "boot_id": self.boot,
                            "session_id": self.session,
                            "request_id": str(uuid.uuid4()),
                            "seq": self.grip_seq,
                            "separate_heartbeat": True,
                            "side": side,
                            "value": float(values[index]),
                            "mode": "fine",
                            "target_generation": f.gripper["target_generation"],
                        }
                        self.record(
                            "gripper_lane_request",
                            request_id=request["request_id"],
                            side=side,
                            value=float(values[index]),
                            observation_time=observed,
                            due=due,
                        )
                    receipt = self.io.gripper_command(request)
                    with self.lock:
                        if self.state != "ACTIVE":
                            raise ControlFault("gripper_ack_after_stop")
                        if (
                            receipt.get("last_request_id") != request["request_id"]
                            or receipt.get("result", {}).get("ok") is not True
                            or not self.owned_gripper(receipt)
                        ):
                            raise ControlFault("gripper_lane_receipt_failed")
                        self.grip_target[index] = values[index]
                        self.record(
                            "gripper_lane_ack",
                            request_id=request["request_id"],
                            side=side,
                            execution_acknowledged=False,
                        )
            except Exception as exc:
                self.async_error = "gripper:" + repr(exc)
                self.trip(self.async_error)
            finally:
                with self.lock:
                    self.gripper_pending = False

    def poll(self):
        if self.async_error and self.state == "ACTIVE":
            self.trip("async_lane_failure:" + self.async_error)
        return super().poll()

    def close(self):
        if self.state == "ACTIVE":
            self.trip("backend_closed")
        self.lane_stop.set()
        self.gripper_thread.join(timeout=2)
        self.audit_thread.join(timeout=2)
        if self.gripper_thread.is_alive() or self.audit_thread.is_alive():
            self.state = "FAULT_UNCONFIRMED"
            self.record_fault("async_lane_shutdown_unconfirmed")
        super().close()
