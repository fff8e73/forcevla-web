"""Legacy gripper-restart URL now explicitly sets both initial position targets."""

import asyncio
import json
import subprocess
import time
import uuid
from pathlib import Path

import numpy as np

from forcevla_web.home import ApexHomeClient
from forcevla_web.manager import ACTIVE, Conflict

BUSY = {"CHECKING", "RESTARTING", "SENDING_INITIAL", "VERIFYING"}
REVIEW = {"UNCERTAIN", "INTERRUPTED_REVIEW"}


class ApexGripperRestartClient(ApexHomeClient):
    operation_name = "夹爪回初始位"

    def reset_initial(self):
        script = Path(__file__).resolve().parents[1] / "scripts/run_gripper_initial.sh"
        process = subprocess.run(
            [str(script), "--execute"], capture_output=True, text=True, timeout=12
        )
        try:
            result = json.loads(process.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError) as exc:
            raise RuntimeError("初始目标发送结果未确认：" + process.stderr[-600:]) from exc
        if (
            process.returncode != 0
            or result.get("status") != "initial_targets_confirmed"
            or result.get("goal_readback_confirmed") is not True
            or result.get("target") != [0.0, 0.0]
        ):
            raise RuntimeError("初始目标下发未确认：" + json.dumps(result, ensure_ascii=False))
        return result


class GripperRestartController:
    def __init__(self, manager, config, backend=None):
        self.manager = manager
        self.enabled = config.get("source") == "live"
        self.backend = backend or ApexGripperRestartClient()
        self.root = Path(config["data_dir"]) / "gripper-restart"
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.task = None
        self.lock = asyncio.Lock()
        self.verify_timeout = 3.0
        path = self.root / "current.json"
        if path.is_file():
            self.current = json.loads(path.read_text())
            if self.current["state"] in BUSY:
                self.update("INTERRUPTED_REVIEW", "服务重启；夹爪操作未自动重试，请核对反馈")

    @property
    def blocked(self):
        return bool(self.current and self.current["state"] in BUSY | REVIEW)

    def status(self, owner=None):
        run = None if not self.current else {k: v for k, v in self.current.items() if k != "owner"}
        if run:
            run["is_owner"] = owner == self.current["owner"]
        return dict(enabled=self.enabled, blocked=self.blocked, run=run)

    def update(self, state, message, **fields):
        self.current.update(state=state, message=message, updated_unix=time.time(), **fields)
        for name in ("current.json", self.current["id"] + ".json"):
            path = self.root / name
            temp = path.with_suffix(".tmp")
            temp.write_text(
                json.dumps(self.current, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            temp.replace(path)

    def preflight(self):
        if not self.enabled:
            raise Conflict("模拟页面不能设置真机夹爪目标")
        if self.manager.closed_loop.blocked:
            raise Conflict("请先停止真机闭环，等待停止确认并释放夹爪")
        if self.manager.control_validation.blocked:
            raise Conflict("请先结束控制验证并释放占用")
        if self.manager.home.blocked:
            raise Conflict("HOME正在执行或状态待确认，请先等待结束")
        if self.manager.run and self.manager.run["state"] in ACTIVE:
            raise Conflict("请先结束当前推理，再设置夹爪初始目标")

    @staticmethod
    def feedback(gripper, after_unix=0.0):
        if gripper.get("reserved") is not False:
            raise RuntimeError("夹爪仍被其他会话占用，请先释放")
        age = gripper.get("status_age_sec")
        stamp = gripper.get("sample_time_ns")
        if (
            gripper.get("available") is not True
            or type(age) not in (int, float)
            or not np.isfinite(age)
            or not 0 <= age <= 0.5
            or type(stamp) not in (int, float)
            or not np.isfinite(stamp)
            or stamp / 1e9 <= after_unix
            or not 0 <= time.time() - stamp / 1e9 <= 0.5
        ):
            raise RuntimeError("等待目标下发后的新鲜夹爪反馈")
        if gripper.get("healthy") is not True or gripper.get("errors") != [1, 1]:
            raise RuntimeError("夹爪电机尚未报告使能、无故障")
        for key in ("actual", "applied_target", "motor_feedback_age"):
            values = np.asarray(gripper.get(key), dtype=float)
            if values.shape != (2,) or not np.isfinite(values).all():
                raise RuntimeError("夹爪反馈无效：" + key)
        ages = np.asarray(gripper["motor_feedback_age"])
        if np.any(ages < 0) or np.any(ages + age > 0.5) or np.any(stamp / 1e9 - ages <= after_unix):
            raise RuntimeError("夹爪电机反馈过期")
        # No target-position tolerance: reset success is not physical arrival.
        return gripper

    async def start(self, owner, command_id):
        async with self.lock:
            self.preflight()
            if self.current and self.current.get("command_id") == command_id:
                if self.current["owner"] != owner:
                    raise Conflict("夹爪复位请求属于其他浏览器")
                return self.status(owner)
            if self.blocked:
                raise Conflict("夹爪正在回初始位或状态待核对")
            self.current = dict(
                id=uuid.uuid4().hex,
                owner=owner,
                command_id=command_id,
                created_unix=time.time(),
                command_attempted=False,
                operation="initial_target_v1",
                initial_target=[0.0, 0.0],
            )
            self.update("CHECKING", "正在核对夹爪初始目标条件")
            self.task = asyncio.create_task(self.execute())
            return self.status(owner)

    async def execute(self):
        try:
            keys = ("robot", "home", "gripper", "replay")
            values = await asyncio.gather(
                *(asyncio.to_thread(self.backend.get, key) for key in keys)
            )
            data = dict(zip(keys, values))
            robot = data["robot"]
            age = robot.get("arm_state_age_sec")
            if (
                not robot.get("control_ready")
                or robot.get("control_error")
                or type(age) not in (int, float)
                or not np.isfinite(age)
                or not 0 <= age <= 1
            ):
                raise RuntimeError("机器人未就绪或状态已过期")
            if robot.get("current_input_mode") != 0:
                raise RuntimeError("请先停止控制并将Input设置为None")
            if data["home"].get("busy"):
                raise RuntimeError("原控制台HOME正在运行")
            if data["replay"].get("is_playing") or data["replay"].get("is_paused"):
                raise RuntimeError("请先结束回放")
            if data["gripper"].get("reserved") is not False:
                raise RuntimeError("夹爪仍被占用，请先释放")
            self.manager.home.check_stationary()
            self.current["before"] = data
            self.update(
                "SENDING_INITIAL",
                "正在向左右夹爪发送初始位置目标0",
                command_attempted=True,
                requested_unix=time.time(),
            )
            receipt = await asyncio.to_thread(self.backend.reset_initial)
            confirmed_at = time.time()
            self.update(
                "VERIFYING",
                "左右初始目标已确认，正在读取实际位置",
                receipt=receipt,
                acknowledged_unix=confirmed_at,
            )
            end = time.monotonic() + self.verify_timeout
            last_error = "等待夹爪反馈"
            while time.monotonic() < end:
                try:
                    value = await asyncio.to_thread(self.backend.get, "gripper")
                    self.feedback(value, confirmed_at)
                    if any(abs(target) > 1e-6 for target in value["applied_target"]):
                        raise RuntimeError("夹爪初始目标读回不是0")
                    self.update(
                        "COMPLETED",
                        "左右初始目标已下发并确认（0）；实际位置已记录",
                        after=value,
                        goal_readback_confirmed=True,
                        physical_arrival_verified=False,
                    )
                    return
                except (OSError, RuntimeError, ValueError, TypeError) as exc:
                    last_error = str(exc)
                    await asyncio.sleep(0.1)
            raise RuntimeError(last_error)
        except Exception as exc:
            uncertain = self.current.get("command_attempted")
            self.update("UNCERTAIN" if uncertain else "FAILED", str(exc) + "；未自动重试")

    async def check(self, owner):
        async with self.lock:
            if not self.enabled:
                raise Conflict("模拟页面不能读取真机夹爪")
            if self.current and self.current["state"] in BUSY:
                return self.status(owner)
            value = await asyncio.to_thread(self.backend.get, "gripper")
            self.feedback(value)
            if self.current and self.current["state"] in REVIEW:
                self.update(
                    "FAILED",
                    "已核对夹爪当前使能、反馈正常；上次目标下发结果仍未确认，未重发指令",
                    after=value,
                )
            return self.status(owner)

    async def close(self):
        if self.task and not self.task.done():
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=2)
            except TimeoutError:
                self.update("INTERRUPTED_REVIEW", "服务结束；夹爪目标下发结果待核对，未自动重试")
