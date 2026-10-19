"""Bounded stop-observe-replan orchestrator, with process-isolated model/IK work."""

import hashlib
import json
import multiprocessing
import os
import signal
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from robot_control.closed_loop_contracts import Barrier, Cancellation, CycleIdentity
from robot_control.closed_loop_observation import ObservationPending
from robot_control.command_bridge import ControlFault

TERMINAL = {
    "COMPLETED_LIMIT",
    "CANCELLED",
    "TIMEOUT",
    "FAILED",
    "TASK_SUCCESS",
    "STOP_UNCONFIRMED",
    "INTERRUPTED_REVIEW",
}


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".tmp")
    with temp.open("w") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.flush()
    os.replace(temp, path)


def compute_worker(conn, mode, config, runtime):
    """One process, one request at a time. Never owns a robot transport."""
    try:
        os.setsid()
        from robot_control.closed_loop_planning import PrefixPreparer

        if mode == "simulate":
            from robot_control.closed_loop_simulation import SyntheticGeometry, SyntheticPolicy

            policy = SyntheticPolicy(
                delay=runtime.get("simulation_rpc_s", 0.05),
                fault=runtime.get("simulation_fault"),
            )
            solver = SyntheticGeometry()
        else:
            from robot_control.closed_loop_live import load_compute

            policy, solver = load_compute(config, runtime)
        preparer = PrefixPreparer(solver, config)
        conn.send(dict(ready=True))
        while True:
            job = conn.recv()
            if job is None:
                break
            job_id, kind, args = job
            try:
                if kind == "infer":
                    value = policy.predict(*args)
                elif kind == "prepare":
                    value = preparer.prepare(*args)
                else:
                    raise ValueError("unknown_compute_operation")
                conn.send(dict(job_id=job_id, ok=True, result=value))
            except Exception as exc:
                conn.send(
                    dict(
                        job_id=job_id,
                        ok=False,
                        error=str(exc),
                        traceback=traceback.format_exc(),
                    )
                )
    except (EOFError, BrokenPipeError):
        pass
    except Exception as exc:
        try:
            conn.send(dict(ready=False, error=str(exc)))
        except (EOFError, BrokenPipeError):
            pass
    finally:
        conn.close()


class ComputeLane:
    def __init__(self, mode, config, runtime):
        ctx = multiprocessing.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(
            target=compute_worker, args=(child, mode, config, runtime), daemon=True
        )
        self.process.start()
        child.close()
        self.active = None

    def ready(self, tick):
        end = time.monotonic() + 20
        while not self.conn.poll(0.01):
            tick()
            if not self.process.is_alive() or time.monotonic() > end:
                raise ControlFault("compute_start_failed")
        result = self.conn.recv()
        if result.get("ready") is not True:
            raise ControlFault(
                "compute_initialization:" + result.get("error", "unknown")
            )

    def call(self, kind, args, timeout, tick, cancel, generation):
        if self.active is not None:
            raise ControlFault("compute_lane_busy")
        self.active = uuid.uuid4().hex
        try:
            self.conn.send((self.active, kind, args))
            end = time.monotonic() + timeout
            while not self.conn.poll(0.005):
                tick()
                cancel.check(generation)
                if not self.process.is_alive():
                    raise ControlFault("compute_process_died")
                if time.monotonic() >= end:
                    raise ControlFault(kind + "_timeout")
            tick()
            cancel.check(generation)
            result = self.conn.recv()
            if result.get("job_id") != self.active:
                raise ControlFault("compute_result_identity_mismatch")
            if not result.get("ok"):
                raise ControlFault(result.get("error", "compute_failed"))
            return result["result"]
        finally:
            self.active = None

    def close(self):
        # Cancelled/expired calls are isolated; their eventual result is never used.
        if self.process.is_alive():
            try:
                if os.getpgid(self.process.pid) == self.process.pid:
                    os.killpg(self.process.pid, signal.SIGTERM)
                else:
                    self.process.terminate()
            except ProcessLookupError:
                pass
            self.process.join(timeout=2)
        if self.process.is_alive():
            self.process.kill()
            self.process.join(timeout=1)
        self.conn.close()


class ClosedLoop:
    def __init__(
        self,
        source,
        session,
        config,
        folder,
        *,
        mode="simulate",
        runtime=None,
        prompt,
        seed=0,
        cancellation=None,
        client_seen=None,
        compute=None,
    ):
        if mode not in ("simulate", "live") or not prompt.strip() or len(prompt) > 2000:
            raise ValueError("Invalid mode or prompt")
        if type(seed) is not int or not 0 <= seed <= 2**63 - 1:
            raise ValueError("Invalid seed")
        self.source, self.session, self.config = source, session, config
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.mode, self.runtime, self.prompt, self.seed = (
            mode,
            runtime or {},
            prompt,
            seed,
        )
        self.cancel = cancellation or Cancellation()
        self.client_seen = client_seen
        self.compute = compute
        self.state = dict(
            id=self.folder.name,
            mode=mode,
            synthetic=mode == "simulate",
            state="IDLE",
            cycle=0,
            completed_cycles=0,
            created_unix=time.time(),
            prompt=prompt,
            seed=seed,
            config=asdict(config),
            config_sha256=config.digest(),
            generation=0,
            reason=None,
            hardware_execution_verified=False,
            history=[],
            execution=None,
        )
        self.events = (self.folder / "events.jsonl").open("x")
        self.started = time.monotonic()
        self.last_save = 0.0
        self.log_bytes = 0

    def emit(self, event, **fields):
        row = dict(
            time=time.monotonic(),
            unix=time.time(),
            event=event,
            cycle=self.state["cycle"],
            **fields,
        )
        data = json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
        self.log_bytes += len(data.encode())
        if self.log_bytes > self.config.max_log_mb * 1024 * 1024:
            raise ControlFault("event_log_quota")
        self.events.write(data)
        self.events.flush()

    def transition(self, state, **fields):
        self.state.update(state=state, **fields)
        self.emit("state", state=state, **fields)
        self.save()

    def save(self):
        if self.state.get("observation_time") is not None:
            self.state["prediction_age_s"] = max(
                0.0, time.monotonic() - self.state["observation_time"]
            )
        self.state.update(
            updated_unix=time.time(),
            generation=self.cancel.generation,
            execution=self.session.status(),
        )
        atomic_json(self.folder / "status.json", self.state)
        self.last_save = time.monotonic()

    def tick(self):
        now = time.monotonic()
        if now - self.started >= self.config.max_task_duration_s:
            self.cancel.cancel("task_timeout")
        if (
            self.client_seen is not None
            and now - self.client_seen() > self.config.browser_disconnect_timeout_s
        ):
            self.cancel.cancel("browser_disconnected")
        self.cancel.check()
        phase = self.session.status()["phase"]
        if phase in ("STOPPING", "STOP_CONFIRMED", "STOP_UNCONFIRMED"):
            raise ControlFault(
                self.session.status().get("reason") or "execution_stopped"
            )
        self.session.supervise()
        if self.state["state"] == "EXECUTE" and phase == "SETTLING":
            self.transition("SETTLING")
        if now - self.last_save > 0.1:
            self.save()

    def wait_io(self, pool, operation, *args):
        job = pool.submit(operation, *args)
        while not job.done():
            self.tick()
            time.sleep(0.005)
        self.tick()
        return job.result()

    def _observe(self, barrier, previous_id):
        end = time.monotonic() + self.config.observation_timeout_s
        last_reason = None
        while True:
            self.tick()
            try:
                return self.source.read_after(barrier, previous_id)
            except ObservationPending as exc:
                last_reason = str(exc)
            if time.monotonic() >= end:
                raise ControlFault("observation_timeout:" + str(last_reason))
            time.sleep(0.01)

    def run(self):
        outcome, reason, previous_id = "FAILED", None, None
        pool = ThreadPoolExecutor(1, thread_name_prefix="loop-operation")
        try:
            self.transition("PRECHECK")
            if self.mode == "live":
                from robot_control.closed_loop_live import runtime_identity

                model_identity = runtime_identity()
            else:
                model_identity = dict(
                    synthetic=True,
                    model="SyntheticPolicy",
                    geometry="SyntheticGeometry",
                )
            atomic_json(
                self.folder / "manifest.json",
                dict(
                    config=asdict(self.config),
                    mode=self.mode,
                    runtime=self.runtime,
                    model_identity=model_identity,
                    source_sha256=self._source_hashes(),
                    created_unix=time.time(),
                ),
            )
            if self.compute is None:
                self.compute = ComputeLane(self.mode, self.config, self.runtime)
            self.compute.ready(self.tick)
            self.transition("WARMING")
            warm = self._observe(Barrier.now(), None)
            warm_id = "warmup-" + uuid.uuid4().hex
            warm_started = time.monotonic()
            warm_result = self.compute.call(
                "infer",
                (warm.request(self.prompt, self.seed), warm_id),
                self.config.warmup_timeout_s,
                self.tick,
                self.cancel,
                self.cancel.generation,
            )
            warm_actions = np.asarray(warm_result.get("actions"))
            if (
                warm_result.get("request_id") != warm_id
                or warm_actions.shape != (1, 50, 14)
                or not np.isfinite(warm_actions).all()
            ):
                raise ControlFault("invalid_warmup_response")
            self.state["warmup_s"] = time.monotonic() - warm_started
            self.emit(
                "warmup_discarded",
                request_id=warm_id,
                observation_id=warm.id,
                seconds=self.state["warmup_s"],
            )
            self.transition("ACQUIRE")
            barrier = self.wait_io(pool, self.session.acquire, self.state["id"])
            for cycle in range(1, self.config.max_cycles + 1):
                if (
                    sum(p.stat().st_size for p in self.folder.rglob("*") if p.is_file())
                    >= self.config.max_log_mb * 1024 * 1024
                ):
                    raise ControlFault("run_storage_quota")
                self.state["cycle"] = cycle
                self.transition("HOLDING", hold=asdict(barrier))
                self.transition("OBSERVE")
                obs = self._observe(barrier, previous_id)
                start_target = self.session.status().get("hold_target")
                if start_target is None:
                    raise ControlFault("missing_confirmed_hold_target")
                obs = replace(
                    obs,
                    metadata={
                        **obs.metadata,
                        "execution_start_target": start_target,
                        "execution_seed_policy": "confirmed_command_target_with_measured_state_check",
                    },
                )
                previous_id = obs.id
                identity = CycleIdentity(
                    self.state["id"],
                    cycle,
                    self.cancel.generation,
                    obs.id,
                    uuid.uuid4().hex,
                )
                folder = self.folder / "cycles" / f"{cycle:04d}"
                folder.mkdir(parents=True, exist_ok=False)
                np.savez_compressed(
                    folder / "observation.npz",
                    state=obs.state,
                    joints=obs.joints,
                    velocities=obs.velocities,
                    gripper_targets=obs.gripper_targets,
                    metadata=json.dumps(obs.metadata),
                    **obs.images,
                )
                record = dict(
                    identity=asdict(identity),
                    hold_before=asdict(barrier),
                    observation_time=obs.observed_at,
                    source_times=obs.source_times,
                    started=time.monotonic(),
                )
                atomic_json(folder / "cycle.json", record)
                self.transition(
                    "INFER",
                    observation_id=obs.id,
                    observation_time=obs.observed_at,
                    request_id=identity.request_id,
                )
                started = time.monotonic()
                prediction = self.compute.call(
                    "infer",
                    (obs.request(self.prompt, self.seed), identity.request_id),
                    self.config.rpc_timeout_s,
                    self.tick,
                    self.cancel,
                    identity.generation,
                )
                record["rpc_s"] = time.monotonic() - started
                if prediction.get("request_id") != identity.request_id:
                    raise ControlFault("model_response_id_mismatch")
                self.transition("PREPARE", rpc_s=record["rpc_s"])
                started = time.monotonic()
                segment = self.compute.call(
                    "prepare",
                    (obs, prediction, identity, folder / "plan"),
                    self.config.prepare_timeout_s,
                    self.tick,
                    self.cancel,
                    identity.generation,
                )
                record["prepare_s"] = time.monotonic() - started
                if (
                    segment.identity != identity
                    or segment.observed_at != obs.observed_at
                ):
                    raise ControlFault("prepared_plan_identity_mismatch")
                self.transition(
                    "EXECUTE", plan_id=segment.digest(), prepare_s=record["prepare_s"]
                )
                result = self.wait_io(pool, self.session.execute, segment, self.cancel)
                barrier = self.wait_io(pool, self.session.wait_hold, self.cancel)
                record.update(
                    result=result, hold_after=asdict(barrier), finished=time.monotonic()
                )
                atomic_json(folder / "cycle.json", record)
                self.state["completed_cycles"] += 1
                self.state["history"].append(
                    dict(
                        cycle=cycle,
                        observation_id=obs.id,
                        plan_id=segment.digest(),
                        rpc_s=record["rpc_s"],
                        prepare_s=record["prepare_s"],
                        hold_confirmed=True,
                        executed_steps=result["executed_model_steps"],
                    )
                )
                self.emit("cycle_complete", observation_id=obs.id, hold=asdict(barrier))
            outcome, reason = "COMPLETED_LIMIT", "cycle_limit"
        except Exception as exc:
            reason = self.cancel.reason or str(exc)
            outcome = (
                "TIMEOUT"
                if reason == "task_timeout"
                else "CANCELLED"
                if reason
                in (
                    "user_cancelled",
                    "browser_disconnected",
                    "parent_disconnected",
                    "service_stopping",
                )
                else "FAILED"
            )
            try:
                self.emit("failure", reason=reason, traceback=traceback.format_exc())
            except Exception:
                pass
        finally:
            self.cancel.cancel(reason or "run_finished")
            # Stop control before terminating a potentially hung compute process.
            try:
                self.state.update(state="STOPPING", reason=reason)
                self.save()
            except Exception:
                pass
            try:
                stop = self.session.stop(reason or "run_finished")
                self.state["stop"] = stop
                if stop.get("confirmed"):
                    self.session.release()
                else:
                    outcome = "STOP_UNCONFIRMED"
            except Exception as exc:
                outcome = "STOP_UNCONFIRMED"
                self.state["stop"] = dict(confirmed=False, error=str(exc))
            if self.compute:
                self.compute.close()
            pool.shutdown(wait=True, cancel_futures=True)
            self.state.update(state=outcome, reason=reason, finished_unix=time.time())
            try:
                self.session.close()
            except Exception as exc:
                self.state["recording_error"] = str(exc)
                if outcome == "COMPLETED_LIMIT":
                    self.state["state"] = "FAILED"
            self.events.close()
            self.save()
        return self.state

    @staticmethod
    def _source_hashes():
        root = Path(__file__).parent
        paths = (
            list(root.glob("*.py"))
            + list((root / "native").glob("*.cpp"))
            + list((root / "native_loop").glob("*.cpp"))
            + [
                root / "build/offline_qp",
                root / "build/loop/loop_motion",
                root / "run_native_loop.sh",
                root / "build_loop.sh",
                root / "native_loop/CMakeLists.txt",
                root / "native_loop/vendor/nlohmann/json.hpp",
                root / "native_loop/vendor/LICENSE.MIT",
            ]
        )
        return {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in paths
            if p.is_file()
        }
