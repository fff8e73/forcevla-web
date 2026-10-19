"""Single-flight prediction inference and generation-based late-result rejection."""

import asyncio
import hashlib
import json
import time
import uuid
from collections import deque
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from forcevla_web.heartbeat import Heartbeats
from forcevla_web.offline_ik import OfflineChecks

ACTIVE = {"WARMING", "RUNNING", "WAITING_INPUT", "STOPPING"}


class Conflict(Exception):
    pass


class Manager:
    def __init__(self, config, source, policy, store):
        self.config, self.source, self.policy, self.store = config, source, policy, store
        self.snapshot = None
        self.input = {"ready": False, "reasons": ["等待采集进程"], "ages": {}}
        self.healthy = False
        self.health_error = "尚未检查"
        self.run = None
        self.latest = None
        self.generation = 0
        self.job = None
        self.tasks = []
        self.last_seen = {}
        self.heartbeats = Heartbeats(config["data_dir"], self.touch)
        self.durations = deque(maxlen=200)
        self.events = deque(maxlen=100)
        self.storage_error = None
        self.config_revision = hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest()[:16]
        self.source_hashes = {
            str(path.relative_to(Path(__file__).parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in Path(__file__).parent.rglob("*")
            if path.suffix in (".py", ".js", ".html", ".css")
        }
        self.ik = OfflineChecks(self)
        self.manifest = {}
        if config.get("manifest_path") and Path(config["manifest_path"]).exists():
            self.manifest = json.loads(Path(config["manifest_path"]).read_text())

    async def launch(self):
        await asyncio.to_thread(self.store.interrupt_pending_diagnostics)
        for run in self.store.runs():
            if run["state"] in ACTIVE:
                run.update(state="INTERRUPTED", reason="服务重启，旧会话未自动恢复")
                self.store.save_run(run)
        self.tasks = [asyncio.create_task(self.collect()), asyncio.create_task(self.health())]
        await self.ik.start()

    async def close(self):
        if self.run and self.run["state"] in ACTIVE:
            await self.stop(self.run["id"], self.run["owner"], "服务关闭")
        if self.job:
            await self.job
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        await self.ik.close()
        self.policy.close()
        self.heartbeats.close()

    async def collect(self):
        while True:
            try:
                self.snapshot = await asyncio.to_thread(self.source.read)
                self.input = (
                    self.snapshot.validate(self.config)
                    if self.snapshot
                    else {"ready": False, "reasons": ["等待采集进程 / IPC文件不存在"], "ages": {}}
                )
            except Exception as exc:
                self.snapshot = None
                self.input = {"ready": False, "reasons": [f"采集读取失败: {exc}"], "ages": {}}
            if self.run and self.run["state"] in ACTIVE - {"STOPPING"}:
                if (
                    time.monotonic() - self.last_seen.get(self.run["owner"], 0)
                    > self.config["disconnect_grace_s"]
                ):
                    await self.stop(self.run["id"], self.run["owner"], "浏览器心跳超时")
            await asyncio.sleep(0.1)

    async def health(self):
        while True:
            try:
                self.healthy = await asyncio.to_thread(self.policy.healthy)
                self.health_error = "" if self.healthy else "模型健康检查未通过"
            except Exception as exc:
                self.healthy, self.health_error = False, str(exc)[:300]
            await asyncio.sleep(5)

    def touch(self, owner):
        self.last_seen[owner] = time.monotonic()
        if len(self.last_seen) > 1000:
            now = time.monotonic()
            self.last_seen = {k: t for k, t in self.last_seen.items() if now - t < 60}

    def emit(self, kind, payload):
        try:
            event = self.store.event(self.run["id"] if self.run else None, kind, payload)
            self.events.append(event)
        except Exception as exc:
            self.storage_error = str(exc)
            raise

    def persist(self):
        try:
            self.store.save_run(self.run)
        except Exception as exc:
            self.storage_error = str(exc)
            raise

    def set_state(self, state, reason=""):
        if self.run["state"] == state and self.run.get("reason") == reason:
            return
        self.run.update(state=state, reason=reason)
        self.persist()
        self.emit("state", {"state": state, "reason": reason, "generation": self.generation})

    async def start(self, options, owner):
        if getattr(self, "gripper_restart", None) and self.gripper_restart.blocked:
            raise Conflict("夹爪正在回初始位或状态待核对")
        if getattr(self, "home", None) and self.home.blocked:
            raise Conflict("HOME正在执行或状态待确认，请先等待复位完成")
        if getattr(self, "closed_loop", None) and self.closed_loop.blocked:
            raise Conflict("分段闭环或未确认的保持尚未结束")
        if getattr(self, "control_validation", None) and self.control_validation.blocked:
            raise Conflict("真实链路验证或设备占用尚未结束，不能开始推理")
        if self.run and self.run.get("command_id") == options["command_id"]:
            if self.run["owner"] != owner:
                raise Conflict("此会话属于其他浏览器")
            return self.public_run(owner)
        if self.job and not self.job.done():
            raise Conflict("已有推理会话或在途请求，请等待其结束")
        if self.storage_error:
            raise Conflict(f"日志存储不可用: {self.storage_error}")
        if not self.healthy:
            raise Conflict("模型服务未就绪")
        current = self.snapshot
        validation = current.validate(self.config) if current else self.input
        if not validation["ready"]:
            raise Conflict("输入未就绪: " + "；".join(validation["reasons"]))
        self.store.check_space()
        self.touch(owner)
        self.generation += 1
        self.latest = None
        self.ik.latest_checked_id = None
        self.durations.clear()
        self.run = {
            "id": uuid.uuid4().hex,
            "owner": owner,
            "state": "WARMING",
            "generation": self.generation,
            "created_unix": time.time(),
            "created_mono": time.monotonic(),
            "count": 0,
            "discarded": 0,
            "reason": "首次请求单独计时",
            "config_revision": self.config_revision,
            "source": self.config["source"],
            "mode": "prediction",
            "config": self.config,
            "deployment_manifest": self.manifest,
            "web_source_sha256": self.source_hashes,
            **options,
        }
        self.persist()
        self.emit("started", {"generation": self.generation, "prompt": options["prompt"]})
        self.job = asyncio.create_task(self.infer(self.generation))
        return self.public_run(owner)

    async def stop(self, run_id, owner, reason="用户停止推理"):
        if not self.run or self.run["id"] != run_id:
            raise Conflict("不是当前会话")
        if self.run["owner"] != owner:
            raise Conflict("仅会话持有者可停止")
        if self.run["state"] in ACTIVE - {"STOPPING"}:
            self.generation += 1
            self.set_state("STOPPING", reason)
        return self.public_run(owner)

    async def infer(self, generation):
        last_id = None
        try:
            while generation == self.generation:
                self.store.check_space()
                snapshot = self.snapshot
                validation = snapshot.validate(self.config) if snapshot else self.input
                if not validation["ready"]:
                    self.set_state("WAITING_INPUT", "；".join(validation["reasons"]))
                    await asyncio.sleep(0.2)
                    continue
                if snapshot.id == last_id:
                    await asyncio.sleep(0.1)
                    continue
                last_id = snapshot.id
                first = self.run["count"] == 0
                self.set_state(
                    "WARMING" if first else "RUNNING", "首次请求单独计时" if first else ""
                )
                request_id = uuid.uuid4().hex
                raw = {**snapshot.images, "state": snapshot.state, "prompt": self.run["prompt"]}
                if self.run["seed"] is not None:
                    raw["seed"] = self.run["seed"]
                self.emit(
                    "request",
                    {
                        "request_id": request_id,
                        "snapshot_id": snapshot.id,
                        "generation": generation,
                    },
                )
                start = time.monotonic()
                output = await asyncio.to_thread(self.policy.predict, raw, request_id)
                elapsed = time.monotonic() - start
                actions = np.asarray(output["actions"])
                normalized = np.asarray(output["normalized_actions"])
                if (
                    actions.shape != (1, 50, 14)
                    or not np.isfinite(actions).all()
                    or normalized.shape != (1, 50, 32)
                    or not np.isfinite(normalized).all()
                ):
                    raise ValueError("模型输出形状或有限值校验失败")
                discarded = generation != self.generation
                act = actions[0]
                displacement = [
                    float(
                        np.linalg.norm(
                            act[:, offset : offset + 3]
                            - snapshot.state[state_offset : state_offset + 3],
                            axis=1,
                        ).max()
                    )
                    for offset, state_offset in ((0, 0), (7, 13))
                ]
                rotation = [
                    float(
                        (
                            Rotation.from_euler("xyz", snapshot.state[s + 3 : s + 6]).inv()
                            * Rotation.from_euler("xyz", act[:, a + 3 : a + 6])
                        )
                        .magnitude()
                        .max()
                    )
                    for a, s in ((0, 0), (7, 13))
                ]
                result = {
                    "id": request_id,
                    "request_id": request_id,
                    "run_id": self.run["id"],
                    "generation": generation,
                    "snapshot_id": snapshot.id,
                    "created_unix": time.time(),
                    "completed_mono": time.monotonic(),
                    "rpc_s": elapsed,
                    "model_s": output["inference_time_us"] / 1e6,
                    "startup_sample": first,
                    "observation_age_s": time.monotonic()
                    - min(s["received_mono"] for s in snapshot.meta["sources"].values()),
                    "actions": act.tolist(),
                    "normalized_actions": normalized[0].tolist(),
                    "state": snapshot.state.tolist(),
                    "input_metadata": snapshot.meta,
                    "input_validation": validation,
                    "discarded": discarded,
                    "max_displacement_m": displacement,
                    "max_rotation_rad": rotation,
                    "gripper_in_range": bool(
                        ((act[:, [6, 13]] >= 0) & (act[:, [6, 13]] <= 1)).all()
                    ),
                    "offline_ik": {
                        "status": "not_requested" if discarded else "pending",
                        "passed": False,
                        "execution_verified": False,
                    }
                    if self.ik.enabled
                    else "unavailable",
                    "execution_verified": False,
                    "step_period_s": 1 / 30,
                }
                await asyncio.to_thread(self.store.prediction, result, snapshot)
                if discarded:
                    self.run["discarded"] += 1
                    self.emit(
                        "discarded", {"request_id": request_id, "reason": "会话已停止或版本已变更"}
                    )
                    break
                self.latest = result
                self.run["count"] += 1
                if not first:
                    self.durations.append(elapsed)
                self.emit(
                    "prediction",
                    {"request_id": request_id, "rpc_s": elapsed, "startup_sample": first},
                )
                self.persist()
                await self.ik.submit(result, snapshot, generation)
                if self.run["kind"] == "single":
                    break
                await asyncio.sleep(0)
            self.set_state("IDLE", "推理已结束；保留历史结果")
        except Exception as exc:
            self.run.update(state="ERROR", reason=str(exc)[:500])
            try:
                self.persist()
                self.emit("error", {"message": self.run["reason"]})
            except Exception:
                pass

    def public_run(self, owner=None):
        if not self.run:
            return None
        return {
            k: v
            for k, v in self.run.items()
            if k not in ("owner", "config", "deployment_manifest", "web_source_sha256")
        } | {"is_owner": self.run["owner"] == owner}

    def status(self, owner=None):
        state = (
            self.snapshot.state
            if self.snapshot is not None and self.snapshot.state.shape == (26,)
            else []
        )
        values = [float(v) if np.isfinite(v) else None for v in state]
        metrics = {"steady_samples": len(self.durations), "p50_s": None, "p95_s": None}
        if self.durations:
            metrics.update(
                p50_s=float(np.percentile(self.durations, 50)),
                p95_s=float(np.percentile(self.durations, 95)),
            )
        return {
            "schema_version": 1,
            "home": self.home.status(owner) if getattr(self, "home", None) else None,
            "gripper_restart": self.gripper_restart.status(owner) if getattr(self, "gripper_restart", None) else None,
            "connection_health": self.heartbeats.summary(owner),
            "closed_loop": self.closed_loop.status(owner)
            if getattr(self, "closed_loop", None)
            else None,
            "control_validation": self.control_validation.status(owner)
            if getattr(self, "control_validation", None)
            else None,
            "mode": "prediction",
            "source": self.config["source"],
            "model": {"healthy": self.healthy, "error": self.health_error, "name": "forcevla-dual"},
            "input": {
                **self.input,
                "image_warning": self.snapshot.meta.get("image_warning") if self.snapshot else None,
            },
            "state": values,
            "snapshot_id": self.snapshot.id if self.snapshot else None,
            "run": self.public_run(owner),
            "metrics": metrics,
            "storage_error": self.storage_error,
            "latest_prediction_id": self.latest["id"] if self.latest else None,
            "latest_age_s": time.monotonic() - self.latest["completed_mono"]
            if self.latest
            else None,
            "capabilities": {
                "hardware_write": False,
                "offline_ik": self.ik.available,
                "live_input": self.config["source"] == "live",
            },
            "ik": self.ik.status(),
            "latest_prediction_revision": self.latest["offline_ik"].get("completed_unix", 0)
            if self.latest and isinstance(self.latest.get("offline_ik"), dict)
            else 0,
            "joint_feedback": self.snapshot.meta.get("joints") if self.snapshot else None,
            "events": list(self.events)[-30:],
        }
