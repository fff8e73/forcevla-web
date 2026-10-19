"""Allowlisted PhyAI model process lifecycle for the web console."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
from pathlib import Path


class ModelManager:
    def __init__(self, config):
        self.config = config
        self.registry = {
            item["id"]: item for item in config.get("model_registry", [])
        }
        self.process = None
        self.selected = None
        self.state = "IDLE"
        self.error = ""
        self.metadata = None
        self.started_at = None
        self._lock = asyncio.Lock()

    def _log_tail(self, limit=1800):
        path = Path(self.config["data_dir"]) / "model-manager.log"
        try:
            return path.read_text(errors="replace")[-limit:]
        except OSError:
            return ""

    def _progress(self):
        text = self._log_tail(12000)
        stages = [
            ("RPC 服务已就绪", 100, "READY"),
            ("Engine ready", 96, "引擎就绪"),
            ("plugin_setup", 78, "加载模型权重/插件"),
            ("stage=kernel", 52, "初始化 CUDA kernel"),
            ("stage=cuda", 32, "初始化 CUDA"),
            ("stage=torch_setup", 18, "初始化 PyTorch"),
        ]
        for marker, percent, label in stages:
            if marker in text:
                return percent, label
        if self.state == "RUNNING":
            return 100, "READY"
        if self.state == "ERROR":
            return 0, "失败"
        return 5, "启动模型进程"

    def public(self):
        process = self.process
        alive = process is not None and process.returncode is None
        if process is not None and not alive and self.state in {"STARTING", "RUNNING"}:
            self.state = "ERROR"
            self.error = f"模型进程已退出，code={process.returncode}"
        percent, stage = self._progress()
        return {
            "state": self.state,
            "selected": self.selected,
            "error": self.error,
            "metadata": self.metadata,
            "pid": process.pid if alive else None,
            "started_at": self.started_at,
            "log_path": str(Path(self.config["data_dir"]) / "model-manager.log"),
            "log_tail": self._log_tail(),
            "progress": percent,
            "stage": stage,
            "elapsed_s": max(0, time.time() - self.started_at) if self.started_at else 0,
            "available": list(self.registry.values()),
        }

    def list_models(self):
        return list(self.registry.values())

    async def _probe(self, endpoint, timeout_s=5.0):
        repo = self.config["phyai_repository"]
        code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from phyai_robot.remote import RemotePolicy
p = RemotePolicy(sys.argv[2], timeout_s=float(sys.argv[3]))
try:
    print(json.dumps(p.info().to_dict(), ensure_ascii=False))
finally:
    p.close()
"""
        result = await asyncio.to_thread(
            subprocess.run,
            [self.config["python"], "-c", code, repo, endpoint, str(timeout_s)],
            capture_output=True,
            text=True,
            timeout=timeout_s + 2,
            env={**os.environ, "PYTHONPATH": repo},
        )
        if result.returncode:
            raise RuntimeError((result.stderr or result.stdout).strip()[-1000:])
        return json.loads(result.stdout)

    async def load(self, model_id):
        async with self._lock:
            if model_id not in self.registry:
                raise ValueError("模型不在允许的部署目录列表中")
            if self.process is not None and self.process.returncode is None:
                raise RuntimeError("已有模型正在运行，请先停止")
            item = self.registry[model_id]
            bundle = Path(item["bundle"])
            if not bundle.is_dir() or not (bundle / "model.sh").is_file():
                raise RuntimeError(f"部署包或 model.sh 不存在: {bundle}")
            config_path = bundle / "deployment.json"
            if not config_path.is_file():
                raise RuntimeError("部署包缺少 deployment.json")
            deployment = json.loads(config_path.read_text())
            endpoint = deployment["endpoint"]
            self.state, self.error, self.metadata = "STARTING", "GPU 初始化/加载权重中", None
            command = ["bash", str(bundle / "model.sh"), "--start-server", "--allow-unvalidated-model"]
            log_path = Path(self.config["data_dir"]) / "model-manager.log"
            log = log_path.open("ab")
            self.process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(bundle),
                stdout=log,
                stderr=asyncio.subprocess.STDOUT,
                start_new_session=True,
            )
            self.selected = dict(item, endpoint=endpoint, deployment=str(config_path))
            self.started_at = time.time()
            deadline = time.monotonic() + self.config.get("model_load_timeout_s", 180)
            try:
                while time.monotonic() < deadline:
                    if self.process.returncode is not None:
                        raise RuntimeError(f"模型进程退出，code={self.process.returncode}")
                    try:
                        self.metadata = await self._probe(endpoint, timeout_s=3)
                        self.state = "RUNNING"
                        self.error = ""
                        return self.public()
                    except Exception as exc:
                        self.error = str(exc)
                        await asyncio.sleep(1)
                raise RuntimeError("等待模型 RPC metadata 超时")
            except Exception:
                await self._stop_locked()
                self.state = "ERROR"
                raise

    async def _stop_locked(self):
        if self.process is None or self.process.returncode is not None:
            return
        try:
            os.killpg(self.process.pid, signal.SIGINT)
            await asyncio.wait_for(self.process.wait(), timeout=15)
        except (ProcessLookupError, asyncio.TimeoutError):
            if self.process.returncode is None:
                os.killpg(self.process.pid, signal.SIGKILL)
                await self.process.wait()

    async def stop(self):
        async with self._lock:
            await self._stop_locked()
            self.process = None
            self.state = "IDLE"
            self.error = ""
            self.metadata = None
            return self.public()

    async def close(self):
        await self.stop()
