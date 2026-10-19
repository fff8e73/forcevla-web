"""Explicit stationary control tests through existing Apex HTTP APIs.

No trajectory/target/SDK writes. Input None and claim/hold/release are the only
mutations. Success confirms stationary interfaces, never moving-stop behavior.
"""

import asyncio
import json
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

import numpy as np

from forcevla_web.manager import ACTIVE, Conflict

BUSY = {"CHECKING", "SELECTING_NONE", "CLAIMING", "OBSERVING", "HOLDING", "RELEASING", "ABORTING"}
REVIEW = {"HELD_REVIEW", "UNCERTAIN", "INTERRUPTED_REVIEW"}


class CheckFailed(RuntimeError):
    pass


class ApexValidationClient:
    def get(self, name):
        paths = {
            "robot": "/api/v1/robot/status",
            "gripper": "/api/v1/gripper/web/status",
            "replay": "/api/v1/replay/status",
        }
        return self._request(paths[name])

    def select_none(self):
        result = self._request("/api/v1/control/set_input", {"input": 0})
        if result.get("success") is not True:
            raise CheckFailed("Input None切换未确认")
        return result

    def gripper(self, action, boot, session, safety_ack=False):
        if action not in {"claim", "hold", "release"}:
            raise ValueError("Target/heartbeat commands are not allowed here")
        request_id = str(uuid.uuid4())
        return self._request(
            "/api/v1/gripper/web/command",
            {
                "action": action,
                "boot_id": boot,
                "request_id": request_id,
                "session_id": session,
                "safety_ack": safety_ack,
                "separate_heartbeat": True,
            },
        )

    @staticmethod
    def _request(path, body=None):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8080" + path,
            data=data,
            headers={"Content-Type": "application/json"},
            method="GET" if data is None else "POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=2.5 if body is not None else 1) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise CheckFailed(
                f"Apex拒绝HTTP {exc.code}: {exc.read(1000).decode(errors='replace')}"
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
            raise CheckFailed("Apex请求未确认，未自动重发：" + str(exc)) from exc
        if result.get("ok") is not True or not isinstance(result.get("data"), dict):
            raise CheckFailed("Apex状态不可用")
        return result["data"]


def telemetry_report(
    snapshot,
    robot,
    gripper,
    replay,
    now=None,
    wall=None,
    allow_owned=False,
    gripper_speed_limit=0.03,
):
    now = time.monotonic() if now is None else now
    wall = time.time() if wall is None else wall
    reasons = []
    measurements = {}
    if not robot.get("control_ready") or robot.get("control_error"):
        reasons.append("机器人未就绪或存在错误")
    age = robot.get("arm_state_age_sec")
    if not isinstance(age, (int, float)) or not 0 <= age <= 0.5:
        reasons.append("机器人状态过期")
    if robot.get("current_input_mode") not in [0, 1]:
        reasons.append("当前来源不是None/Teleop，不能中断Planner或Playback")
    if robot.get("current_mode") not in [1, 3]:
        reasons.append("Robot Mode需为Position或Impedance；本测试不修改Robot Mode")
    if replay.get("is_playing") or replay.get("is_paused"):
        reasons.append("请先结束回放")
    if not gripper.get("available") or not gripper.get("healthy"):
        reasons.append("夹爪反馈不可用或不健康")
    if gripper.get("reserved") and not allow_owned:
        reasons.append("夹爪已被其他会话占用")
    status_age = gripper.get("status_age_sec")
    if not isinstance(status_age, (int, float)) or not 0 <= status_age <= 0.5:
        reasons.append("夹爪状态过期")
    cf = snapshot.meta.get("control_feedback", {}) if snapshot else {}
    for key in ["joints", "left_gripper", "right_gripper"]:
        item = cf.get(key, {})
        received = item.get("received_mono")
        if (
            item.get("error")
            or not isinstance(received, (int, float))
            or not 0 <= now - received <= 0.5
        ):
            reasons.append(key + ": 实时位置/速度缺失或过期")
            continue
        if key == "joints":
            try:
                q = np.asarray(item["positions"], float)
                v = np.asarray(item["velocities"], float)
                if (
                    q.shape != (14,)
                    or v.shape != (14,)
                    or not np.isfinite(q).all()
                    or not np.isfinite(v).all()
                ):
                    raise ValueError()
                if not -0.05 <= wall - item["source_time"] <= 0.5:
                    raise ValueError()
                measurements["joint_positions_rad"] = q.tolist()
                measurements["joint_max_speed_rad_s"] = float(np.max(np.abs(v)))
                if np.any(np.abs(v) > 0.02):
                    reasons.append("双臂尚未静止（关节速度>0.02 rad/s）")
            except (KeyError, ValueError, TypeError):
                reasons.append("关节位置/速度或源时间无效")
        else:
            try:
                q = float(item["position_rad"])
                v = float(item["velocity_rad_s"])
                if not np.isfinite([q, v]).all():
                    raise ValueError()
                measurements[key] = {
                    "position_rad": q,
                    "velocity_rad_s": v,
                    "age_s": now - received,
                }
                if abs(v) > gripper_speed_limit:
                    reasons.append(f"{key}: 网页速度检查超限（>{gripper_speed_limit:g} rad/s）")
            except (KeyError, ValueError, TypeError):
                reasons.append(key + ": 无效电机反馈")
    try:
        calibration = np.asarray(gripper["calibration"], float)
        targets = np.asarray(gripper["applied_target"], float)
        ages = np.asarray(gripper["motor_feedback_age"], float)
        if (
            calibration.shape != (2, 2)
            or targets.shape != (2,)
            or ages.shape != (2,)
            or not np.isfinite(calibration).all()
            or not np.isfinite(targets).all()
            or not np.isfinite(ages).all()
        ):
            raise ValueError()
        if (
            np.any(calibration[:, 1] <= calibration[:, 0])
            or np.any(targets < 0)
            or np.any(targets > 1)
        ):
            raise ValueError()
        if np.any(ages < 0) or status_age is None or np.any(ages + status_age > 0.3):
            reasons.append("底层夹爪反馈过期")
        for i, key in enumerate(["left_gripper", "right_gripper"]):
            if key in measurements:
                target = calibration[i, 0] + targets[i] * (calibration[i, 1] - calibration[i, 0])
                if abs(measurements[key]["position_rad"] - target) > 0.05:
                    reasons.append(key + ": 实际位置与原目标偏差>0.05 rad")
        measurements["gripper_existing_targets"] = targets.tolist()
    except (KeyError, ValueError, TypeError):
        reasons.append("夹爪原目标、标定或反馈年龄无效")
    return {
        "ready": not reasons,
        "reasons": reasons,
        "measurements": measurements,
        "robot_mode": robot.get("current_mode"),
        "input_mode": robot.get("current_input_mode"),
        "actual_gripper": gripper.get("actual"),
        "target_gripper": gripper.get("applied_target"),
        "checked_unix": wall,
        "web_gripper_speed_limit_rad_s": gripper_speed_limit,
        "driver_claim_speed_limit_rad_s": 0.02,
        "scope": "stationary conditions only, not movement-stop proof",
    }


class ControlValidation:
    def __init__(self, manager, config, backend=None):
        self.manager, self.config = manager, config
        self.backend = backend or ApexValidationClient()
        self.enabled = (
            config.get("source") == "live"
            and config.get("control_validation", {}).get("enabled") is True
        )
        self.root = Path(config["data_dir"]) / "control-validation"
        self.root.mkdir(exist_ok=True)
        self.current = None
        self.job = None
        self.aborted = False
        self.last_check = None
        self.operation_lock = asyncio.Lock()
        self.period = config.get("control_validation", {}).get("poll_period_s", 0.15)
        self.gripper_speed_limit = config.get("control_validation", {}).get(
            "gripper_speed_limit_rad_s", 0.03
        )
        if (
            type(self.gripper_speed_limit) not in (int, float)
            or not np.isfinite(self.gripper_speed_limit)
            or not 0 < self.gripper_speed_limit <= 0.03
        ):
            raise ValueError("Web gripper speed limit must be in (0,0.03] rad/s")
        path = self.root / "current.json"
        if path.exists():
            self.current = json.loads(path.read_text())
            if self.current["state"] in BUSY:
                self.current.update(
                    state="INTERRUPTED_REVIEW",
                    reason="服务重启，未自动恢复或释放设备；需要人工检查",
                )
                self.save()

    @property
    def blocked(self):
        return bool(self.current and self.current["state"] in BUSY | REVIEW)

    def save(self):
        if self.current:
            self.current["updated_unix"] = time.time()
            text = json.dumps(self.current, ensure_ascii=False, indent=2, allow_nan=False)
            for name in ["current.json", self.current["id"] + ".json"]:
                p = self.root / name
                temp = p.with_suffix(".tmp")
                temp.write_text(text + "\n")
                temp.replace(p)

    def event(self, state, message, **data):
        self.current.update(state=state, reason=message)
        self.current["events"].append(
            {"time_unix": time.time(), "state": state, "message": message, **data}
        )
        self.save()

    def status(self, owner):
        run = self.current
        public = (
            None
            if run is None
            else {
                k: v for k, v in run.items() if k not in ["owner", "driver_session", "driver_boot"]
            }
        )
        if public:
            public["is_owner"] = run["owner"] == owner
        return {
            "enabled": self.enabled,
            "run": public,
            "last_check": self.last_check,
            "blocked": self.blocked,
            "stationary_test_only": True,
            "web_gripper_speed_limit_rad_s": self.gripper_speed_limit,
            "driver_claim_speed_limit_rad_s": 0.02,
            "trajectory_execution_enabled": False,
            "moving_stop_verified": False,
            "finish_input_mode": "None（不自动恢复Teleop）",
        }

    async def read(self):
        robot, gripper, replay = await asyncio.gather(
            *(asyncio.to_thread(self.backend.get, k) for k in ["robot", "gripper", "replay"])
        )
        return robot, gripper, replay

    async def check(self):
        if not self.enabled:
            raise Conflict("真实链路验证仅在已启用的live页面可用")
        robot, gripper, replay = await self.read()
        self.last_check = telemetry_report(
            self.manager.snapshot,
            robot,
            gripper,
            replay,
            gripper_speed_limit=self.gripper_speed_limit,
        )
        return self.last_check

    async def start(self, owner, options):
        async with self.operation_lock:
            if getattr(self.manager, "gripper_restart", None) and self.manager.gripper_restart.blocked:
                raise Conflict("夹爪正在回初始位或状态待核对")
            if getattr(self.manager, "home", None) and self.manager.home.blocked:
                raise Conflict("HOME正在执行或状态待确认")
            if getattr(self.manager, "closed_loop", None) and self.manager.closed_loop.blocked:
                raise Conflict("分段闭环或未确认的保持尚未结束")
            if not self.enabled:
                raise Conflict("真实链路验证未启用")
            if self.current and self.current.get("command_id") == options["command_id"]:
                if self.current["owner"] != owner:
                    raise Conflict("此操作属于其他会话")
                return self.status(owner)
            if self.blocked:
                raise Conflict("已有验证或未释放会话，请先检查/处理")
            if self.manager.run and self.manager.run["state"] in ACTIVE:
                raise Conflict("请先停止影子推理，等待在途请求结束")
            if options.get("safety_ack") is not True:
                raise Conflict("请先确认现场条件和Input None切换")
            self.manager.store.check_space()
            self.manager.touch(owner)
            self.aborted = False
            self.current = {
                "id": uuid.uuid4().hex,
                "owner": owner,
                "command_id": options["command_id"],
                "kind": options["kind"],
                "state": "CHECKING",
                "reason": "检查静止条件",
                "created_unix": time.time(),
                "events": [],
                "driver_session": str(uuid.uuid4()),
                "driver_boot": None,
                "claim_attempted": False,
                "input_none_attempted": False,
                "post_count": 0,
                "web_gripper_speed_limit_rad_s": self.gripper_speed_limit,
                "target_commands_sent": 0,
                "moving_stop_verified": False,
                "mode_restored": False,
            }
            self.save()
            self.job = asyncio.create_task(self.execute())
            return self.status(owner)

    def watch(self):
        if self.aborted:
            raise CheckFailed("用户中止验证")
        if time.monotonic() - self.manager.last_seen.get(self.current["owner"], 0) > 5:
            raise CheckFailed("验证持有者连接已断开")

    async def capture(self, allow_owned=False, baseline=None):
        self.watch()
        robot, gripper, replay = await self.read()
        check = telemetry_report(
            self.manager.snapshot,
            robot,
            gripper,
            replay,
            allow_owned=allow_owned,
            gripper_speed_limit=self.gripper_speed_limit,
        )
        self.last_check = check
        if not check["ready"]:
            raise CheckFailed("；".join(check["reasons"]))
        if baseline:
            if robot.get("current_mode") != baseline["robot_mode"]:
                raise CheckFailed("Robot Mode在验证中发生变化")
            m = check["measurements"]
            b = baseline["measurements"]
            if (
                np.max(np.abs(np.array(m["joint_positions_rad"]) - b["joint_positions_rad"]))
                > 0.005
            ):
                raise CheckFailed("观察到关节变化>0.005 rad，中止静止验证")
            for side in ["left_gripper", "right_gripper"]:
                if abs(m[side]["position_rad"] - b[side]["position_rad"]) > 0.01:
                    raise CheckFailed("观察到夹爪位置变化>0.01 rad")
            if not np.allclose(
                m["gripper_existing_targets"], b["gripper_existing_targets"], atol=1e-6, rtol=0
            ):
                raise CheckFailed("原夹爪目标发生变化")
        return check, robot, gripper

    async def post_gripper(self, action):
        self.current["post_count"] += 1
        self.save()
        return await asyncio.to_thread(
            self.backend.gripper,
            action,
            self.current["driver_boot"],
            self.current["driver_session"],
            True,
        )

    def owned(self, gripper):
        return (
            gripper.get("boot_id") == self.current["driver_boot"]
            and gripper.get("session_id") == self.current["driver_session"]
            and gripper.get("reserved") is True
        )

    async def execute(self):
        try:
            baseline, robot, gripper = await self.capture()
            self.current["initial"] = baseline
            self.current["driver_boot"] = gripper["boot_id"]
            self.save()
            for _ in range(3):
                await asyncio.sleep(self.period)
                await self.capture(baseline=baseline)
            self.event("SELECTING_NONE", "仅切换输入来源为None，Robot Mode保持不变")
            self.current["input_none_attempted"] = True
            self.current["post_count"] += 1
            self.save()
            await asyncio.to_thread(self.backend.select_none)
            check, robot, gripper = await self.capture(baseline=baseline)
            if robot.get("current_input_mode") != 0:
                raise CheckFailed("Input None读回未确认")
            if robot.get("current_mode") != baseline["robot_mode"]:
                raise CheckFailed("Robot Mode变化")
            if gripper.get("boot_id") != self.current["driver_boot"]:
                raise CheckFailed("夹爪驱动已重启")
            self.event("CLAIMING", "继承现有夹爪目标接管，不发送新目标")
            self.current["claim_attempted"] = True
            self.save()
            claimed = await self.post_gripper("claim")
            if not self.owned(claimed):
                raise CheckFailed("接管会话读回不匹配")
            if not np.allclose(
                claimed.get("issued"),
                baseline["measurements"]["gripper_existing_targets"],
                atol=1e-6,
                rtol=0,
            ):
                raise CheckFailed("接管初始目标变化")
            self.event("OBSERVING", "真实会话已接管，观察位置/速度和现有目标")
            deadline = time.monotonic() + (3.2 if self.current["kind"] == "lease_expiry" else 0.5)
            seen_held = False
            while time.monotonic() < deadline:
                await asyncio.sleep(self.period)
                check, robot, gripper = await self.capture(allow_owned=True, baseline=baseline)
                if robot.get("current_input_mode") != 0 or not self.owned(gripper):
                    raise CheckFailed("来源或接管所有权变化")
                if gripper.get("state") == "held":
                    seen_held = True
                self.current["latest_feedback"] = check
                self.save()
            if self.current["kind"] == "lease_expiry":
                if not seen_held or "心跳" not in gripper.get("reason", ""):
                    raise CheckFailed("租约超时保持未确认")
                self.current["lease_hold_observed"] = True
            self.event("HOLDING", "发送保持请求并检查驱动held回执；这不是运动中停止证明")
            held = await self.post_gripper("hold")
            if not self.owned(held) or held.get("state") != "held":
                raise CheckFailed("保持未确认")
            self.current["hold_acknowledged"] = True
            self.save()
            await asyncio.sleep(self.period)
            await self.capture(allow_owned=True, baseline=baseline)
            self.watch()
            self.event("RELEASING", "释放本验证会话，Input仍保持None")
            released = await self.post_gripper("release")
            if released.get("reserved") is not False or released.get("session_id") is not None:
                raise CheckFailed("释放未确认")
            final, robot, gripper = await self.capture(baseline=baseline)
            if robot.get("current_input_mode") != 0:
                raise CheckFailed("结束时Input不是None")
            self.current["final"] = final
            self.current["release_acknowledged"] = True
            self.event(
                "COMPLETED_STATIONARY",
                "静止接管/保持/释放验证完成；未发送运动目标，运动中停止尚未验证",
            )
        except Exception as exc:
            self.current["error"] = str(exc)
            await self.hold_after_failure(str(exc))

    async def hold_after_failure(self, reason):
        if not self.current.get("claim_attempted"):
            self.event("BLOCKED", reason)
            return
        try:
            grip = await asyncio.to_thread(self.backend.get, "gripper")
            if self.owned(grip):
                held = await self.post_gripper("hold")
                if not self.owned(held) or held.get("state") != "held":
                    raise CheckFailed("保持读回未确认")
                self.event("HELD_REVIEW", reason + "；本会话已保持，仍保留占用，请检查后明确释放")
            elif (
                grip.get("boot_id") == self.current["driver_boot"] and grip.get("reserved") is False
            ):
                self.event("ABORTED_UNOWNED", reason + "；读回显示本会话未占用")
            else:
                self.event("UNCERTAIN", reason + "；驱动重启/所有权变化，未操作其他会话")
        except Exception as exc:
            self.event("UNCERTAIN", reason + "；无法确认保持：" + str(exc))

    async def abort(self, owner):
        if not self.current or self.current["owner"] != owner:
            raise Conflict("仅验证持有者可中止")
        self.aborted = True
        if self.current["state"] in BUSY:
            self.event("ABORTING", "已请求中止，正在确认本会话保持状态")
        return self.status(owner)

    async def release(self, owner, safety_ack):
        async with self.operation_lock:
            if not self.current or self.current["owner"] != owner:
                raise Conflict("仅验证持有者可释放")
            if not safety_ack:
                raise Conflict("请明确确认释放")
            if self.job and not self.job.done():
                raise Conflict("等待在途操作结束后再释放")
            if self.current["state"] not in REVIEW:
                raise Conflict("没有待处理占用")
            self.manager.touch(owner)
            self.aborted = False
            check, robot, grip = await self.capture(allow_owned=True)
            if robot.get("current_input_mode") != 0 or not self.owned(grip):
                raise Conflict("不是本验证会话或Input不是None，请在Apex检查")
            held = await self.post_gripper("hold")
            if not self.owned(held) or held.get("state") != "held":
                raise Conflict("保持未确认")
            released = await self.post_gripper("release")
            if released.get("reserved") is not False:
                raise Conflict("释放未确认")
            self.event("RELEASED", "本验证会话已明确释放，Input保持None")
            return self.status(owner)

    async def close(self):
        self.aborted = True
        if self.job and not self.job.done():
            await self.job
