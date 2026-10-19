"""Web control of an isolated bounded closed-loop worker, never robot IO here."""

import asyncio
import hashlib
import json
import os
import re
import signal
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

from forcevla_web.manager import ACTIVE, Conflict

TERMINAL = {
    "COMPLETED_LIMIT",
    "CANCELLED",
    "TIMEOUT",
    "FAILED",
    "TASK_SUCCESS",
    "STOP_UNCONFIRMED",
    "INTERRUPTED_REVIEW",
}
REVIEW = {"STOP_UNCONFIRMED", "INTERRUPTED_REVIEW"}


def save(path, value):
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2))
    temp.replace(path)


def process_signature(pid):
    try:
        stat = Path(f"/proc/{int(pid)}/stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, ValueError, IndexError):
        return None


class ClosedLoopController:
    def __init__(self, manager, config):
        self.manager, self.config = manager, config
        self.options = config.get("closed_loop", {})
        self.enabled = self.options.get("enabled", False)
        self.root = Path(config["data_dir"]) / "closed-loop"
        self.root.mkdir(parents=True, exist_ok=True)
        os.chmod(self.root, 0o700)
        self.integration = Path(
            self.options.get("integration_root", str(Path(__file__).resolve().parents[1] / "robot_control"))
        )
        self.config_root = Path(self.options.get("config_root", self.integration / "config" if (self.integration / "config").is_dir() else self.integration.parent / "config"))
        self.profile_path = Path(
            self.options.get("profile_path", self.config_root / "closed_loop_v1.json")
        )
        self.profile = json.loads(self.profile_path.read_text()) if self.enabled else {}
        self.planner_profile_path = self.config_root / "planner_loop.json"
        self.planner_profile = (json.loads(self.planner_profile_path.read_text())
                                if self.planner_profile_path.is_file() else {})
        self.current = None
        self.process = None
        self.task = None
        self.lock = asyncio.Lock()
        self.closing = False
        path = self.root / "current.json"
        if path.exists():
            self.current = json.loads(path.read_text())

    def _alive(self):
        if not self.current:
            return False
        if self.process is not None:
            return self.process.returncode is None
        pid = self.current.get("pid")
        return bool(
            pid
            and self.current.get("process_signature")
            and process_signature(pid) == self.current["process_signature"]
        )

    def read_run(self, run_id, owner=None):
        if not re.fullmatch(r"[a-f0-9]{32}", run_id):
            raise KeyError(run_id)
        folder = self.root / run_id
        meta = json.loads((folder / "web.json").read_text())
        path = folder / "status.json"
        value = (
            json.loads(path.read_text())
            if path.exists()
            else dict(id=run_id, state="STARTING", completed_cycles=0, mode=meta["options"]["mode"])
        )
        current = self.current and self.current["id"] == run_id
        if current and not self._alive() and value["state"] not in TERMINAL:
            # A missing process handle alone is never reported as a confirmed stop.
            value.update(state="INTERRUPTED_REVIEW", reason="工作进程中断；保持状态需复核")
        return dict(value, is_owner=meta["owner"] == owner, created_unix=meta["created_unix"])

    @property
    def blocked(self):
        if not self.current:
            return False
        try:
            state = self.read_run(self.current["id"])["state"]
            return self._alive() or state not in TERMINAL or state in REVIEW
        except (OSError, ValueError, KeyError):
            return True

    def status(self, owner=None):
        blockers = []
        qualification = self.options.get("qualification_path")
        if not qualification or not Path(qualification).is_file():
            blockers.append("待完成单段运动、停止与时序验收")
        if self.config.get("source") != "live":
            blockers.append("当前页面为模拟输入")
        try:
            run = self.read_run(self.current["id"], owner) if self.current else None
        except (OSError, ValueError, KeyError):
            run = dict(state="INTERRUPTED_REVIEW", reason="运行记录不可读", is_owner=False)
        return dict(
            enabled=self.enabled,
            profile=self.profile,
            run=run,
            blocked=self.blocked,
            simulation_available=self.enabled,
            live_available=self.enabled and not blockers,
            live_blockers=blockers,
            planner_available=(self.enabled and self.config.get("source") == "live"
                               and self.options.get("planner_enabled") is True
                               and bool(self.planner_profile)
                               and (self.integration / "run_planner_loop.py").is_file()),
            planner_profile=self.planner_profile,
        )

    async def start(self, owner, options):
        async with self.lock:
            if getattr(self.manager, "gripper_restart", None) and self.manager.gripper_restart.blocked:
                raise Conflict("夹爪正在回初始位或状态待核对")
            if getattr(self.manager, "home", None) and self.manager.home.blocked:
                raise Conflict("HOME正在执行或状态待确认，请先等待复位完成")
            if not self.enabled:
                raise Conflict("分段闭环未启用")
            digest = hashlib.sha256(json.dumps(options, sort_keys=True).encode()).hexdigest()
            for folder in self.root.iterdir():
                meta_path = folder / "web.json"
                if not folder.is_dir() or not meta_path.is_file():
                    continue
                meta = json.loads(meta_path.read_text())
                if meta.get("command_id") == options["command_id"]:
                    if meta["owner"] != owner or meta["request_digest"] != digest:
                        raise Conflict("重复请求标识对应其他会话或不同参数")
                    return self.read_run(folder.name, owner)
            if self.blocked:
                raise Conflict("已有闭环或保持未确认的会话")
            if self.manager.run and self.manager.run["state"] in ACTIVE:
                raise Conflict("请先结束影子推理，等待在途请求完成")
            if self.manager.control_validation.blocked:
                raise Conflict("静止控制验证或占用尚未结束")
            planner = options.get("engine", "native") == "movej"
            if planner:
                if not self.status(owner)["planner_available"]:
                    raise Conflict("真机闭环尚未启用")
                if options["mode"] != "live" or type(options["execution_horizon"]) is not int or not 1 <= options["execution_horizon"] <= 50:
                    raise Conflict("真机MoveJ闭环每轮执行步数必须为1～50的整数")
                if not self.manager.healthy or not self.manager.input.get("ready"):
                    raise Conflict("模型或当前观察尚未就绪")
            if options["mode"] == "live" and not planner:
                status = self.status(owner)
                if not status["live_available"]:
                    raise Conflict("；".join(status["live_blockers"]))
                if self.config.get("source") != "live":
                    raise Conflict("模拟页面不能启动真机")
            self.manager.store.check_space()
            run_id = uuid.uuid4().hex
            folder = self.root / run_id
            folder.mkdir(mode=0o700)
            profile = dict(
                self.planner_profile if planner else self.profile,
                execution_horizon=options["execution_horizon"],
                max_cycles=options["max_cycles"],
                max_task_duration_s=options["max_task_duration_s"],
            )
            save(folder / "profile.json", profile)
            runtime = dict(
                web_root=str(Path(__file__).resolve().parents[1]),
                ipc_path=self.config.get("ipc_path"),
                policy_repo=self.config.get("policy_repo"),
                model_address=self.config.get("model_address"),
                planning_config=str(self.config_root / "planning.json"),
                qualification_path=self.options.get("qualification_path"),
                simulation_rpc_s=self.options.get("simulation_rpc_s", 0.05),
                simulation_fault=self.options.get("simulation_fault"),
                simulation_freeze_after=self.options.get("simulation_freeze_after"),
            )
            save(folder / "runtime.json", runtime)
            meta = dict(
                id=run_id,
                owner=owner,
                created_unix=time.time(),
                command_id=options["command_id"],
                request_digest=digest,
                options=options,
            )
            save(folder / "web.json", meta)
            self.current = dict(meta)
            save(self.root / "current.json", self.current)
            self.manager.touch(owner)
            log = (folder / "worker.log").open("ab")
            try:
                self.process = await asyncio.create_subprocess_exec(
                    str(self.integration / "run.sh"),
                    "planner-loop" if planner else "closed-loop",
                    "--mode",
                    options["mode"],
                    "--config",
                    str(folder / "profile.json"),
                    "--output",
                    str(folder),
                    "--runtime",
                    str(folder / "runtime.json"),
                    "--prompt",
                    options["prompt"],
                    "--seed",
                    str(options["seed"]),
                    "--parent-control",
                    cwd=str(self.integration),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=log,
                    stderr=log,
                    env={
                        **os.environ,
                        "OPENBLAS_NUM_THREADS": "1",
                        "OMP_NUM_THREADS": "1",
                        "PYTHONUNBUFFERED": "1",
                    },
                )
            except Exception as exc:
                save(
                    folder / "status.json",
                    dict(
                        id=run_id,
                        state="FAILED",
                        reason="启动失败: " + str(exc),
                        mode=options["mode"],
                        completed_cycles=0,
                    ),
                )
                raise Conflict("闭环进程启动失败") from exc
            finally:
                log.close()
            self.current.update(
                pid=self.process.pid, process_signature=process_signature(self.process.pid)
            )
            save(self.root / "current.json", self.current)
            self.task = asyncio.create_task(self._supervise())
            return self.read_run(run_id, owner)

    async def _send(self, command):
        if not self.process or self.process.returncode is not None or not self.process.stdin:
            return
        try:
            self.process.stdin.write((json.dumps(command) + "\n").encode())
            await asyncio.wait_for(self.process.stdin.drain(), timeout=0.25)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass

    async def _supervise(self):
        process = self.process
        owner = self.current["owner"]
        waiter = asyncio.create_task(process.wait())
        while process.returncode is None and not self.closing:
            seen = self.manager.last_seen.get(owner, 0)
            await self._send(dict(command="heartbeat", seen_monotonic=seen,
                                  connection_health=self.manager.heartbeats.summary(owner)))
            try:
                await asyncio.wait_for(asyncio.shield(waiter), timeout=0.2)
            except TimeoutError:
                pass
        if not self.closing:
            await process.wait()

    async def stop(self, run_id, owner):
        async with self.lock:
            value = self.read_run(run_id, owner)
            if not value["is_owner"]:
                raise Conflict("此闭环属于其他浏览器会话")
            if not self.current or self.current["id"] != run_id:
                return value
            if value["state"] in TERMINAL and not self._alive():
                return value
            if self.process and self.process.returncode is None:
                await self._send(dict(command="stop"))
            elif self._alive():
                os.kill(self.current["pid"], signal.SIGTERM)
            return dict(value, stop_requested=True)

    def runs(self, owner):
        rows = []
        for folder in self.root.iterdir():
            if folder.is_dir() and (folder / "web.json").is_file():
                try:
                    row = self.read_run(folder.name, owner)
                    rows.append(
                        {
                            k: row.get(k)
                            for k in (
                                "id",
                                "state",
                                "created_unix",
                                "mode",
                                "completed_cycles",
                                "reason",
                                "is_owner",
                            )
                        }
                    )
                except (OSError, ValueError, KeyError):
                    continue
        return sorted(rows, key=lambda r: r["created_unix"], reverse=True)[:100]

    def export(self, run_id, owner):
        value = self.read_run(run_id, owner)
        if value["state"] not in TERMINAL:
            raise Conflict("请在运行结束后导出完整记录")
        output = tempfile.TemporaryFile()
        try:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                folder = self.root / run_id
                for path in sorted(folder.rglob("*")):
                    if (
                        path.is_file()
                        and not path.is_symlink()
                        and path.name not in ("web.json", "runtime.json")
                    ):
                        archive.write(path, path.relative_to(folder))
            output.seek(0)
        except Exception:
            output.close()
            raise
        return output

    async def close(self):
        self.closing = True
        if self.process and self.process.returncode is None:
            await self._send(dict(command="stop"))
            if self.process.stdin:
                self.process.stdin.close()
            try:
                await asyncio.wait_for(self.process.wait(), timeout=5)
            except TimeoutError:
                # SIGTERM is handled by the worker as a stop request. Never turn
                # an absent reply into a claim that the robot is stopped.
                self.process.terminate()
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
