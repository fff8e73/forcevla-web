"""Private native timing process and adapters; RPC/IK stay in other processes."""

import json
import socket
import struct
import subprocess
import threading
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from robot_control.closed_loop import atomic_json
from robot_control.closed_loop_execution import ContinuousSession
from robot_control.closed_loop_live import RosMotionTransport
from robot_control.closed_loop_simulation import SimulatedIO
from robot_control.command_bridge import ControlFault

ROOT = Path(__file__).parent
BINARY = ROOT / "build/loop/loop_motion"


class NativeClient:
    def __init__(
        self, config, folder, *, simulated, namespace="", domain=231, qualification=None
    ):
        if not BINARY.is_file():
            raise ControlFault("native_executor_not_built")
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        profile = self.folder / "native-profile.json"
        if profile.exists():
            raise ControlFault("native_output_reuse_forbidden")
        atomic_json(profile, asdict(config))
        self.lock = threading.Lock()
        self.state_lock = threading.Lock()
        self.snapshot = None
        self.error = None
        self.seen = time.monotonic()
        self.finished = threading.Event()
        self.socket, child = socket.socketpair()
        self.socket.settimeout(0.2)
        command = [
            str(ROOT / "run_native_loop.sh"),
            "--fd",
            str(child.fileno()),
            "--profile",
            str(profile),
            "--output",
            str(self.folder),
            "--domain",
            str(domain),
            "--namespace",
            namespace,
        ]
        if simulated:
            command.append("--simulate")
        elif qualification:
            command.extend(["--qualification", str(qualification)])
        self.log = (self.folder / "native-process.log").open("wb")
        self.process = subprocess.Popen(
            command,
            pass_fds=(child.fileno(),),
            stdout=self.log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        child.close()
        # ROS initialization can exceed a normal IPC deadline; this happens before
        # control is acquired and is explicitly bounded.
        self.socket.settimeout(10.0)
        try:
            result = self.command("status")
            deadline = time.monotonic() + 5
            while not result.get("timer_tid") and time.monotonic() < deadline:
                time.sleep(0.01)
                result = self.command("status")
            if not result.get("timer_tid"):
                raise ControlFault("native_timer_initialization_timeout")
            atomic_json(
                self.folder / "scheduling.json",
                dict(
                    method="native_thread_existing_RealtimeKit_policy",
                    requested_priority=10,
                    rtkit_result=result.get("rtkit_result"),
                    timer_policy=result.get("timer_policy"),
                    timer_tid=result.get("timer_tid"),
                    timer_cpu=result.get("timer_cpu"),
                    rttime_limit_us=200000,
                ),
            )
        except Exception:
            self.socket.close()
            self.process.terminate()
            self.process.wait(timeout=2)
            self.log.close()
            raise
        self.socket.settimeout(0.2)
        self.thread = threading.Thread(
            target=self._poll, name="native-executor-status", daemon=True
        )
        self.thread.start()

    def _receive(self, n):
        data = bytearray()
        while len(data) < n:
            block = self.socket.recv(n - len(data))
            if not block:
                raise ControlFault("native_executor_disconnected")
            data.extend(block)
        return bytes(data)

    def command(self, op, **values):
        with self.lock:
            if self.error:
                raise ControlFault(self.error)
            request = json.dumps(
                dict(op=op, supervised_at=self.seen, **values), allow_nan=False
            ).encode()
            try:
                self.socket.sendall(struct.pack("!I", len(request)) + request)
                size = struct.unpack("!I", self._receive(4))[0]
                if size > 4 * 1024 * 1024:
                    raise ControlFault("native_response_oversized")
                response = json.loads(self._receive(size))
            except (OSError, ValueError, ControlFault) as exc:
                self.error = "native_IPC:" + str(exc)
                raise ControlFault(self.error) from exc
            if not response.get("ok"):
                raise ControlFault(response.get("error", "native_rejected"))
            with self.state_lock:
                self.snapshot = response["result"]
                self.received = time.monotonic()
            return response["result"]

    def status(self):
        with self.state_lock:
            if (
                self.error
                or self.snapshot is None
                or time.monotonic() - self.received > 0.2
            ):
                raise ControlFault(self.error or "native_status_stale")
            return dict(self.snapshot)

    def supervise(self, seen):
        self.seen = max(self.seen, seen)

    def _poll(self):
        while not self.finished.wait(0.004):
            try:
                self.command("status")
            except Exception as exc:
                self.error = str(exc)
                return

    def close(self):
        self.finished.set()
        self.thread.join(timeout=0.5)
        try:
            self.command("close")
        except Exception:
            pass
        self.socket.close()
        try:
            self.process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=3)
        self.log.close()
        report_path = self.folder / "native-result.json"
        if self.process.returncode != 0 or not report_path.is_file():
            raise ControlFault("native_exit_or_report_unconfirmed")
        if json.loads(report_path.read_text()).get("recording_failed"):
            raise ControlFault("native_journal_failure")


class NativeTransportMixin:
    has_native_timing = True

    def native_status(self):
        return self.native.status()

    def supervise(self, seen):
        self.native.supervise(seen)

    def abort_native(self, reason):
        return self.native.command("abort", reason=reason)

    def arm_segment(self, segment):
        # Parent-side geometry and plan authorization is retained for real ROS IO.
        super().arm_segment(segment)
        result = self.native.command(
            "arm",
            id=segment.digest(),
            seed=segment.seed.tolist(),
            seed_velocity=segment.seed_velocity.tolist(),
            observed_at=segment.observed_at,
            q=segment.joints.tolist(),
            brakes=[b.tolist() for b in segment.brake_paths],
            test_fail_side=(
                0
                if self.fail_side == "left"
                else 1
                if self.fail_side == "right"
                else -1
            )
            if isinstance(self, SimulatedIO)
            else -1,
        )
        self.motion_started = result["started"]

    def bind_stationary_target(self, q, grip):
        super().bind_stationary_target(q, grip)
        status = self.gripper_status()
        self.native.command(
            "bind",
            q=np.asarray(q).tolist(),
            bounds=self.native_bounds.tolist(),
            boot=status["boot_id"],
            session=status["session_id"],
        )

    def select_input(self, value):
        super().select_input(value)
        if value == 3:
            end = time.monotonic() + 0.4
            while True:
                try:
                    self.native.command("activate")
                    break
                except ControlFault as exc:
                    if "feedback_pending" not in str(exc) or time.monotonic() > end:
                        raise
                    time.sleep(0.01)
        elif value == 0 and getattr(self, "native", None):
            # Idle/deactivated native state sends no targets. A moving session
            # must finish its checked stop before releasing the source.
            status = self.native.status()
            if status["state"] not in ("EXECUTE", "BRAKING"):
                self.native.command("deactivate")

    def publish_arm(self, side, positions):
        if side == "left":
            self._native_pair = np.empty(14)
            self._native_pair[:7] = positions
        elif side == "right" and self._native_pair is not None:
            self._native_pair[7:] = positions
            if isinstance(self, SimulatedIO) and self.fail_side:
                raise ControlFault("injected_arm_failure:" + self.fail_side)
            self.native.command("hold", q=self._native_pair.tolist())
            self._native_pair = None
        else:
            raise ControlFault("native_hold_pair_order")

    def close(self):
        try:
            self.native.close()
        finally:
            super().close()


class NativeSimulatedIO(NativeTransportMixin, SimulatedIO):
    def __init__(self, config, folder, bounds):
        super().__init__()
        self.native_bounds = np.array(bounds, copy=True)
        self._native_pair = None
        self.native = NativeClient(config, folder, simulated=True)

    def feedback(self):
        status = self.native.status()
        with self.lock:
            self.q = np.array(status["measured_q"], dtype=float)
            self.target = np.array(status["last_q"], dtype=float)
            result = super().feedback()
            return replace(
                result,
                received_at=status["feedback_time"],
                target_received_at=status["feedback_time"],
            )


class NativeRosMotionTransport(NativeTransportMixin, RosMotionTransport):
    def __init__(
        self,
        config,
        folder,
        bounds,
        qualification_path=None,
        *,
        domain=24,
        namespace="/tj",
    ):
        self.native_bounds = np.array(bounds, copy=True)
        self._native_pair = None
        # Qualification and explicit endpoints are validated by the base before
        # either process obtains a production publisher.
        super().__init__(
            config,
            qualification_path,
            domain=domain,
            namespace=namespace,
            create_arm_publishers=False,
        )
        try:
            self.native = NativeClient(
                config,
                folder,
                simulated=False,
                namespace=namespace,
                domain=domain,
                qualification=qualification_path,
            )
        except Exception:
            RosMotionTransport.close(self)
            raise

    def validate_targets(self, q, grip):
        # Normal holds and stop holds are exactly the native process's last
        # submitted pair. Arm segments use its independent whole-path validation.
        status = self.native.status()
        if np.max(np.abs(np.asarray(q) - np.asarray(status["last_q"]))) > 1e-10:
            raise ControlFault("native_hold_target_mismatch")
        if not np.isfinite(grip).all():
            raise ControlFault("invalid_gripper_target")


class NativeContinuousSession(ContinuousSession):
    def supervise(self):
        super().supervise()
        self.io.supervise(self.last_supervision)

    def _start_stop(self, reason, *, allow_brake=True):
        if self.phase in ("STOPPING", "STOP_CONFIRMED", "STOP_UNCONFIRMED", "RELEASED"):
            return
        if self.stop_reason is None:
            self.stop_reason = reason
        if self.cancellation:
            self.cancellation.cancel(reason)
        if self.phase in ("EXECUTE", "BRAKING"):
            try:
                self.io.abort_native(reason)
                self.phase = "BRAKING"
            except Exception:
                self._terminal()
        else:
            # A claim already in flight must resolve under the backend lock.
            # Latching the sidecar before bind would prevent confirming a hold
            # for a successfully claimed gripper during that race.
            if self.phase != "ACQUIRING":
                try:
                    self.io.abort_native(reason)
                except Exception:
                    pass
            self._terminal()
        self.changed.notify_all()

    def _run(self):
        self.last_gripper_step = -1
        self.native_plan = None
        while not self.finished.wait(0.003):
            try:
                with self.changed:
                    if self.backend.state != "ACTIVE" or self.phase not in (
                        "EXECUTE",
                        "BRAKING",
                        "SETTLING",
                        "HOLDING",
                    ):
                        continue
                    now = time.monotonic()
                    status = self.io.native_status()
                    if status.get("recording_failed"):
                        raise ControlFault("native_journal_failure")
                    if self.audit.error:
                        raise ControlFault("journal_failure:" + self.audit.error)
                    self.backend.last_q = np.array(status["last_q"], dtype=float)
                    if hasattr(self.io, "last_applied"):
                        self.io.last_applied = self.backend.last_q.copy()
                    if now - self.last_supervision > self.config.supervision_timeout_s:
                        self._start_stop("supervisor_expired")
                    if self.gripper_error:
                        self._start_stop("gripper:" + self.gripper_error)
                    if self.cancellation and self.cancellation.event.is_set():
                        self._start_stop(self.cancellation.reason)
                    if (
                        now - self.last_beat >= self.config.heartbeat_period_s
                        and now - self.last_supervision
                        <= self.config.supervision_timeout_s
                    ):
                        self.backend.heartbeat(self.backend.token)
                        self.last_beat = now
                    if self.phase in ("EXECUTE", "BRAKING"):
                        self.index = int(status["sent_ticks"])
                        self.submitted_ticks = int(status["trace_count"])
                        if self.segment is not None and self.index:
                            # Only complete model intervals count as executed.
                            self.executed_steps = sum(
                                1
                                for step in range(self.config.execution_horizon)
                                if self.index
                                >= int(
                                    np.ceil(
                                        (step + 1) / 30 / self.config.control_period_s
                                        - 1e-9
                                    )
                                )
                            )
                        if status["state"] == "ABORTED":
                            self.stop_reason = (
                                self.stop_reason or status["reason"] or "native_stopped"
                            )
                            self._terminal()
                            continue
                        if self.phase == "EXECUTE":
                            p = self.segment
                            if p.identity.request_id != self.native_plan:
                                self.native_plan = p.identity.request_id
                                self.last_gripper_step = -1
                            elapsed = now - self.io.motion_started
                            step = min(
                                int(
                                    max(0, elapsed - self.config.control_period_s) * 30
                                ),
                                self.config.execution_horizon - 1,
                            )
                            if (
                                elapsed >= self.config.control_period_s
                                and step != self.last_gripper_step
                            ):
                                if step != self.last_gripper_step + 1:
                                    raise ControlFault("gripper_schedule_missed")
                                index = int(np.ceil(step / 30 / p.dt - 1e-9))
                                values = p.grippers[index]
                                if not np.array_equal(values, self.backend.grip_target):
                                    if self.gripper_pending:
                                        raise ControlFault("gripper_lane_overrun")
                                    self.gripper_pending = True
                                    self.gripper_queue.put_nowait(
                                        (values.copy(), p.identity.generation)
                                    )
                                self.last_gripper_step = step
                            if status["state"] == "HOLDING" and self.index == len(
                                p.joints
                            ):
                                self._settling()
                    elif status["state"] == "ABORTED":
                        self.stop_reason = self.stop_reason or status["reason"]
                        self._terminal()
                    elif self.phase in ("SETTLING", "HOLDING") and now >= self.next_due:
                        self._normal_hold(now)
                        self.next_due = now + 0.01
            except Exception as exc:
                with self.changed:
                    self.stop_reason = self.stop_reason or str(exc)
                    self._start_stop(str(exc))
                    # A dead native process cannot finish a requested brake.
                    if getattr(self.io.native, "error", None):
                        self._terminal()


def create_session(io, bounds, config, folder):
    return (
        NativeContinuousSession
        if getattr(io, "has_native_timing", False)
        else ContinuousSession
    )(io, bounds, config, folder)
