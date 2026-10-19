"""Isolated, latest-only offline diagnostics. No robot transport dependencies."""

import asyncio
import hashlib
import json
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from forcevla_web.observation import JOINT_NAMES


def seed_request(snapshot, config):
    joint = snapshot.meta.get("joints", {})
    if joint.get("error"):
        raise ValueError(str(joint["error"]))
    if tuple(joint.get("names", [])) != JOINT_NAMES:
        raise ValueError("缺少该观察对应的14关节反馈；不会使用当前关节角替代历史数据")
    q = np.asarray(joint.get("positions"), dtype=float)
    if q.shape != (14,) or not np.isfinite(q).all():
        raise ValueError("关节反馈维度或有限值无效")
    source = joint.get("source") or {}
    image = snapshot.meta.get("sources", {}).get("image", {})
    try:
        age = float(snapshot.meta["created_mono"]) - float(source["received_mono"])
        skew = abs(float(source["source_time"]) - float(image["source_time"]))
        source_age = float(snapshot.meta["created_unix"]) - float(source["source_time"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("缺少关节与图像的对应时间") from exc
    if (
        not np.isfinite([age, skew, source_age]).all()
        or not 0 <= age <= config["max_age_s"]
        or skew > config["max_skew_s"]
        or not -0.1 <= source_age <= config["max_source_age_s"]
    ):
        raise ValueError("该观察的关节反馈过期或与图像不同步")
    return {
        "seed": q.tolist(),
        "observed_tcp": [snapshot.state[:6].tolist(), snapshot.state[13:19].tolist()],
    }, {
        "snapshot_id": snapshot.id,
        "source": source,
        "age_at_snapshot_s": age,
        "image_skew_s": skew,
    }


class NativeWorker:
    def __init__(self, settings):
        self.settings = settings
        self.process = None
        self.metadata = {}

    async def start(self):
        executable = Path(self.settings["worker_path"])
        config_path = Path(self.settings["config_path"])
        if not executable.is_file():
            raise RuntimeError("离线求解程序尚未构建，请运行 scripts/build_ik.sh")
        self.process = await asyncio.create_subprocess_exec(
            str(executable),
            str(config_path),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
            limit=4 * 1024 * 1024,
        )
        hello = await self.read(10)
        if not hello.get("ready"):
            raise RuntimeError(hello.get("error", "离线求解器启动失败"))
        self.metadata = hello["metadata"]
        cfg = json.loads(config_path.read_text())
        paths = [
            executable,
            config_path,
            *(
                Path(cfg[k])
                for k in (
                    "model_path",
                    "tasks_path",
                    "collision_pairs_path",
                    "velocity_limits_path",
                )
            ),
        ]
        paths.append(Path(self.settings["core_library_path"]))
        model_path = Path(cfg["model_path"])
        for mesh in ET.parse(model_path).getroot().findall("./asset/mesh"):
            if "file" in mesh.attrib:
                paths.append((model_path.parent / mesh.attrib["file"]).resolve())
        self.metadata["sha256"] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        }
        self.metadata["config_revision"] = hashlib.sha256(
            json.dumps(self.metadata["sha256"], sort_keys=True).encode()
        ).hexdigest()[:16]
        return self.metadata

    async def read(self, timeout):
        line = await asyncio.wait_for(self.process.stdout.readline(), timeout)
        if not line:
            raise RuntimeError("离线求解进程退出")
        return json.loads(line)

    async def analyze(self, request):
        if not self.process or self.process.returncode is not None:
            raise RuntimeError("离线求解器不可用")
        self.process.stdin.write((json.dumps(request, allow_nan=False) + "\n").encode())
        await self.process.stdin.drain()
        result = await self.read(self.settings.get("timeout_s", 5.0))
        if result.get("request_id") != request["request_id"]:
            raise RuntimeError("离线检查响应ID不匹配")
        return result

    async def close(self):
        if self.process and self.process.returncode is None:
            self.process.terminate()
            try:
                await asyncio.wait_for(self.process.wait(), 2)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
        self.process = None


class OfflineChecks:
    def __init__(self, manager, worker=None):
        self.manager = manager
        settings = manager.config.get("offline_ik", {})
        self.enabled = bool(settings.get("enabled") and manager.config["source"] == "live")
        self.worker = worker or NativeWorker(settings)
        self.available = False
        self.error = ""
        self.metadata = {}
        self.queue = asyncio.Queue(maxsize=1)
        self.task = None
        self.active_id = None
        self.latest_checked_id = None

    async def start(self):
        if not self.enabled:
            return
        try:
            self.metadata = await self.worker.start()
            self.available = True
            self.task = asyncio.create_task(self.loop())
        except Exception as exc:
            self.error = str(exc)
            await self.worker.close()

    async def save(self, prediction_id, report, generation):
        report.update(completed_unix=time.time(), execution_verified=False)
        report["historical"] = generation != self.manager.generation
        await asyncio.to_thread(self.manager.store.update_diagnostic, prediction_id, report)
        if (
            generation == self.manager.generation
            and self.manager.latest
            and self.manager.latest["id"] == prediction_id
        ):
            self.manager.latest["offline_ik"] = report
        if generation == self.manager.generation and report.get("status") not in (
            "pending",
            "superseded",
        ):
            self.latest_checked_id = prediction_id
        stored = await asyncio.to_thread(self.manager.store.get_prediction, prediction_id)
        if stored:
            self.manager.store.event(
                stored["run_id"],
                "offline_ik",
                {
                    "prediction_id": prediction_id,
                    "status": report["status"],
                    "passed": report.get("passed", False),
                    "historical": report["historical"],
                },
            )

    async def submit(self, prediction, snapshot, generation):
        if not self.enabled:
            return
        if not self.available:
            await self.save(
                prediction["id"],
                {"status": "unavailable", "passed": False, "error": self.error},
                generation,
            )
            return
        try:
            data, context = seed_request(snapshot, self.manager.config)
        except ValueError as exc:
            await self.save(
                prediction["id"],
                {"status": "seed_unavailable", "passed": False, "error": str(exc)},
                generation,
            )
            return
        data.update(actions=prediction["actions"], request_id=prediction["id"])
        job = (prediction["id"], generation, data, context)
        if self.queue.full():
            old, old_gen, _, _ = self.queue.get_nowait()
            self.queue.task_done()
            await self.save(
                old,
                {"status": "superseded", "passed": False, "error": "等待中的检查已被更新预测替代"},
                old_gen,
            )
        self.queue.put_nowait(job)

    async def loop(self):
        while True:
            prediction_id, generation, request, context = await self.queue.get()
            self.active_id = prediction_id
            try:
                if generation != self.manager.generation:
                    await self.save(
                        prediction_id,
                        {"status": "superseded", "passed": False, "error": "原会话已停止"},
                        generation,
                    )
                    continue
                started = time.monotonic()
                result = await self.worker.analyze(request)
                result.setdefault("elapsed_s", time.monotonic() - started)
                result["seed_context"] = context
                result["model"] = self.metadata
                await self.save(prediction_id, result, generation)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self.available = False
                self.error = str(exc)
                await self.worker.close()
                try:
                    await self.save(
                        prediction_id,
                        {"status": "error", "passed": False, "error": self.error},
                        generation,
                    )
                except Exception as storage_exc:
                    self.manager.storage_error = str(storage_exc)
                while not self.queue.empty():
                    stale, gen, _, _ = self.queue.get_nowait()
                    self.queue.task_done()
                    try:
                        await self.save(
                            stale,
                            {"status": "unavailable", "passed": False, "error": self.error},
                            gen,
                        )
                    except Exception as storage_exc:
                        self.manager.storage_error = str(storage_exc)
                return
            finally:
                self.active_id = None
                self.queue.task_done()

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await self.worker.close()

    def status(self):
        return {
            "enabled": self.enabled,
            "available": self.available,
            "error": self.error,
            "active_prediction_id": self.active_id,
            "pending": self.queue.qsize(),
            "latest_checked_prediction_id": self.latest_checked_id,
            "model": self.metadata,
        }
