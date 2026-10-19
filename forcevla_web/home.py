"""Native Apex HOME from the inference page. No SDK or joint target publisher."""

import asyncio
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from forcevla_web.manager import ACTIVE, Conflict

BUSY = {"CHECKING", "SELECTING_PLANNER", "MOVING", "RETURNING_NONE"}
REVIEW = {"UNCERTAIN", "INTERRUPTED_REVIEW"}


class ApexHomeClient:
    operation_name = "HOME"

    def request(self, path, body=None, timeout=3):
        request = urllib.request.Request(
            "http://127.0.0.1:8080/api/v1/" + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="GET" if body is None else "POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                f"Apex {self.operation_name}接口拒绝（{exc.code}）：" + exc.read(1000).decode(errors="replace")
            ) from exc
        if result.get("ok") is not True or not isinstance(result.get("data"), dict):
            raise RuntimeError(f"Apex {self.operation_name}接口返回无效")
        return result["data"]

    def get(self, name):
        paths = dict(
            robot="robot/status",
            home="robot/home/status",
            gripper="gripper/web/status",
            replay="replay/status",
        )
        return self.request(paths[name])

    def select(self, index):
        if index not in (0, 2):
            raise ValueError("HOME only selects None or Planner")
        value = self.request("control/set_input", {"input": index})
        if value.get("success") is not True:
            raise RuntimeError("HOME输入来源切换未确认")
        return value

    def home(self):
        # Same bounded, feedback-confirmed operation used by the 8081 console.
        value = self.request("robot/reset", {}, timeout=100)
        if value.get("success") is not True:
            raise RuntimeError(value.get("message", "HOME到位未确认"))
        return value


class HomeController:
    def __init__(self, manager, config, backend=None):
        self.manager = manager
        self.enabled = config.get("source") == "live"
        self.backend = backend or ApexHomeClient()
        self.root = Path(config["data_dir"]) / "home"
        self.root.mkdir(parents=True, exist_ok=True)
        self.current = None
        self.task = None
        self.lock = asyncio.Lock()
        path = self.root / "current.json"
        if path.is_file():
            self.current = json.loads(path.read_text())
            if self.current["state"] in BUSY:
                self.update("INTERRUPTED_REVIEW", "服务重启；需核对原生HOME状态")

    @property
    def blocked(self):
        return bool(self.current and self.current["state"] in BUSY | REVIEW)

    def status(self, owner=None):
        run = None if not self.current else {k: v for k, v in self.current.items() if k != "owner"}
        if run:
            run["is_owner"] = self.current["owner"] == owner
        return dict(enabled=self.enabled, blocked=self.blocked, run=run)

    def update(self, state, message, **fields):
        self.current.update(state=state, message=message, updated_unix=time.time(), **fields)
        for name in ("current.json", self.current["id"] + ".json"):
            path = self.root / name
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(self.current, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
            )
            tmp.replace(path)

    def local_preflight(self):
        if getattr(self.manager, "gripper_restart", None) and self.manager.gripper_restart.blocked:
            raise Conflict("夹爪正在回初始位或状态待核对，请等待结束")
        if not self.enabled:
            raise Conflict("模拟页面不能执行真机HOME")
        if self.manager.closed_loop.blocked:
            raise Conflict("请先停止真机闭环，等待停止确认后再执行HOME")
        if self.manager.control_validation.blocked:
            raise Conflict("请先结束控制验证并释放占用")
        if self.manager.run and self.manager.run["state"] in ACTIVE:
            raise Conflict("请先停止当前推理，再执行HOME")

    async def read(self):
        values = await asyncio.gather(
            *(
                asyncio.to_thread(self.backend.get, key)
                for key in ("robot", "home", "gripper", "replay")
            )
        )
        return dict(zip(("robot", "home", "gripper", "replay"), values))

    @staticmethod
    def check_native(data):
        robot, home, gripper, replay = (data[key] for key in ("robot", "home", "gripper", "replay"))
        if home.get("busy"):
            raise RuntimeError("原控制台HOME正在运行")
        if not robot.get("control_ready") or robot.get("control_error"):
            raise RuntimeError("机器人未就绪或存在错误")
        age = robot.get("arm_state_age_sec")
        if type(age) not in (int, float) or not np.isfinite(age) or not 0 <= age <= 1:
            raise RuntimeError("机器人状态已过期")
        if robot.get("current_mode") not in (1, 3) or robot.get("arm_state") not in (
            [1, 1],
            [3, 3],
        ):
            raise RuntimeError("HOME需要Position或Impedance模式")
        if robot.get("current_input_mode") not in (0, 2):
            raise RuntimeError("请先停止遥操作/回放，将Input设置为None或Planner")
        if gripper.get("reserved") is not False:
            raise RuntimeError("夹爪仍被其他会话占用，请先释放")
        if replay.get("is_playing") or replay.get("is_paused"):
            raise RuntimeError("请先结束轨迹回放")

    def check_stationary(self):
        snap = self.manager.snapshot
        sample = snap.meta.get("control_feedback", {}).get("joints", {}) if snap else {}
        received = sample.get("received_mono")
        velocities = np.asarray(sample.get("velocities"), dtype=float)
        if (
            type(received) not in (int, float)
            or not np.isfinite(received)
            or not 0 <= time.monotonic() - received <= 0.5
            or velocities.shape != (14,)
            or not np.isfinite(velocities).all()
        ):
            raise RuntimeError("等待新鲜的关节速度反馈后再执行HOME")
        # Same stationary threshold as the installed native HOME preflight.
        if np.max(np.abs(velocities)) > 0.02:
            raise RuntimeError("双臂仍在运动，等待静止后再执行HOME")

    async def select_and_confirm(self, index):
        if index == 0:
            current = await asyncio.to_thread(self.backend.get, "robot")
            if current.get("current_input_mode") not in (0, 2):
                raise RuntimeError("控制来源已被其他操作改变，不覆盖其他来源")
        await asyncio.to_thread(self.backend.select, index)
        deadline = time.monotonic() + 2
        while True:
            robot = await asyncio.to_thread(self.backend.get, "robot")
            if robot.get("current_input_mode") == index:
                return robot
            if time.monotonic() > deadline:
                raise RuntimeError("HOME输入来源读回超时")
            await asyncio.sleep(0.05)

    async def start(self, owner, command_id):
        async with self.lock:
            self.local_preflight()
            if self.current and self.current.get("command_id") == command_id:
                if self.current["owner"] != owner:
                    raise Conflict("HOME请求属于其他浏览器")
                return self.status(owner)
            if self.blocked:
                raise Conflict("HOME正在执行或停止状态待确认")
            self.current = dict(
                id=uuid.uuid4().hex,
                owner=owner,
                command_id=command_id,
                created_unix=time.time(),
                selection_attempted=False,
                home_requested=False,
            )
            self.update("CHECKING", "正在核对HOME启动条件")
            self.task = asyncio.create_task(self.execute())
            return self.status(owner)

    async def execute(self):
        try:
            data = await self.read()
            self.check_native(data)
            self.check_stationary()
            self.current["initial"] = data
            self.current["selection_attempted"] = True
            self.update("SELECTING_PLANNER", "正在准备HOME复位")
            await self.select_and_confirm(2)
            self.check_stationary()
            self.update("MOVING", "双臂正在回HOME，等待实际到位确认", home_requested=True)
            native = await asyncio.to_thread(self.backend.home)
            self.update("RETURNING_NONE", "HOME已到位，正在切回Input=None", native_result=native)
            robot = await self.select_and_confirm(0)
            self.update("COMPLETED", "HOME复位完成，Input=None，可继续推理", final_robot=robot)
        except Exception as exc:
            reason = str(exc)
            if not self.current.get("selection_attempted"):
                self.update("FAILED", reason)
                return
            try:
                data = await self.read()
                if data["home"].get("busy"):
                    raise RuntimeError("原生HOME仍在运行，停止尚未确认")
                selected = data["robot"].get("current_input_mode")
                if selected == 2:
                    robot = await self.select_and_confirm(0)
                elif selected == 0:
                    robot = data["robot"]
                else:
                    raise RuntimeError("控制来源已被其他操作改变")
                self.check_stationary()
                self.update("FAILED", reason + "；双臂静止、Input已确认None", final_robot=robot)
            except Exception as cleanup:
                self.update("UNCERTAIN", reason + "；" + str(cleanup))

    async def check(self, owner):
        """Read-only reconciliation after interrupted/uncertain HOME; no motion retry."""
        async with self.lock:
            if self.current and self.current["state"] in BUSY:
                return self.status(owner)
            data = await self.read()
            if self.current and self.current["state"] in REVIEW:
                if data["home"].get("busy") or data["robot"].get("current_input_mode") != 0:
                    raise Conflict("原生HOME或输入来源仍需处理，请在原控制台核对")
                self.check_stationary()
                self.update(
                    "FAILED", "已核对原生HOME结束、双臂静止且Input=None；没有自动重试", review=data
                )
            return self.status(owner)

    async def close(self):
        if self.task and not self.task.done():
            # A detached HTTP request is not proof that native motion stopped.
            # Native HOME keeps its own bounded wait/stop; the journal survives restart.
            try:
                await asyncio.wait_for(asyncio.shield(self.task), timeout=3)
            except TimeoutError:
                self.update("INTERRUPTED_REVIEW", "网页服务结束；原生HOME状态需核对")
