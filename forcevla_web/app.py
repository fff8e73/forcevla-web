import asyncio
import hashlib
import hmac
import json
import os
import secrets
import tempfile
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator
from starlette.middleware.trustedhost import TrustedHostMiddleware

from forcevla_web.closed_loop import ClosedLoopController
from forcevla_web.control_validation import CheckFailed, ControlValidation
from forcevla_web.gripper_restart import GripperRestartController
from forcevla_web.home import HomeController
from forcevla_web.manager import Conflict, Manager
from forcevla_web.model_manager import ModelManager
from forcevla_web.observation import CAMERAS, DemoSource, FileSource, Snapshot
from forcevla_web.policy import DemoPolicy, RemotePolicy
from forcevla_web.store import Store

STATIC = Path(__file__).parent / "static"


class RunOptions(BaseModel):
    kind: Literal["single", "continuous"]
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int | None = Field(default=None, ge=0, le=2**63 - 1, strict=True)
    command_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")


class ValidationOptions(BaseModel):
    kind: Literal["stationary_cycle", "lease_expiry"]
    command_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    safety_ack: bool = Field(strict=True)


class ClosedLoopOptions(BaseModel):
    engine: Literal["native", "movej"] = "native"
    mode: Literal["simulate", "live"]
    prompt: str = Field(min_length=1, max_length=2000)
    seed: int = Field(default=0, ge=0, le=2**63 - 1, strict=True)
    execution_horizon: int = Field(default=3, ge=1, le=50, strict=True)
    max_cycles: int = Field(default=3, ge=1, le=100, strict=True)
    max_task_duration_s: float = Field(default=30, gt=0, le=600, allow_inf_nan=False, strict=True)
    command_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def apply_movej_defaults(cls, values):
        if isinstance(values, dict) and values.get("engine") == "movej":
            values = dict(values)
            values.setdefault("execution_horizon", 20)
            values.setdefault("max_cycles", 50)
            values.setdefault("max_task_duration_s", 600.0)
        return values

    @model_validator(mode="after")
    def check_execution_mode(self):
        if self.engine == "native" and self.execution_horizon > 3:
            raise ValueError("Native mode supports 1..3 steps")
        if self.engine == "movej" and self.mode != "live":
            raise ValueError("MoveJ mode requires live execution")
        return self


class HomeOptions(BaseModel):
    command_id: str = Field(min_length=8, max_length=100, pattern=r"^[a-zA-Z0-9_-]+$")
    model_config = {"extra": "forbid"}


class ConnectionCloseReport(BaseModel):
    connection_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    code: int = Field(ge=0, le=4999, strict=True)
    reason: str = Field(default="", max_length=160)
    was_clean: bool = Field(strict=True)
    visibility_state: Literal["visible", "hidden", "prerender", "unknown"] = "unknown"
    has_focus: bool | None = None
    model_config = {"extra": "forbid"}


class ReleaseOptions(BaseModel):
    safety_ack: bool = Field(strict=True)


def create_app(
    config, manager=None, validation_backend=None, home_backend=None, gripper_restart_backend=None
):
    if manager is None:
        source = DemoSource() if config["source"] == "demo" else FileSource(config["ipc_path"])
        policy = DemoPolicy() if config["source"] == "demo" else RemotePolicy(config)
        manager = Manager(config, source, policy, Store(config["data_dir"], config["quota_mb"]))
    model_manager = ModelManager(config)
    validation = ControlValidation(manager, config, validation_backend)
    manager.control_validation = validation
    closed_loop = ClosedLoopController(manager, config)
    manager.closed_loop = closed_loop
    home = HomeController(manager, config, home_backend)
    manager.home = home
    gripper_restart = GripperRestartController(manager, config, gripper_restart_backend)
    manager.gripper_restart = gripper_restart
    secret_path = Path(config["data_dir"]) / "session.key"
    try:
        fd = os.open(secret_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        secret = secret_path.read_bytes()
    else:
        secret = secrets.token_bytes(32)
        with os.fdopen(fd, "wb") as key_file:
            key_file.write(secret)
    if len(secret) != 32:
        raise RuntimeError(
            "Invalid session key; restore it or remove it while the server is stopped"
        )

    def sign(token):
        return hmac.new(secret, token.encode(), hashlib.sha256).hexdigest()

    def identity(conn):
        value = conn.cookies.get("forcevla_session") or conn.cookies.get("shadow_session", "")
        token, _, signature = value.partition(".")
        if not token or not hmac.compare_digest(signature, sign(token)):
            raise HTTPException(401, "请刷新页面建立会话")
        return token

    def same_origin(conn):
        origin = conn.headers.get("origin", "")
        expected_scheme = "https" if conn.url.scheme in ("https", "wss") else "http"
        parsed = urlsplit(origin)
        return parsed.scheme == expected_scheme and parsed.netloc == conn.headers.get("host")

    def command_owner(request):
        owner = identity(request)
        if not same_origin(request) or not hmac.compare_digest(
            request.headers.get("x-csrf-token", ""), sign("csrf:" + owner)
        ):
            raise HTTPException(403, "会话或来源校验失败")
        return owner

    @asynccontextmanager
    async def lifespan(app):
        await manager.launch()
        yield
        await gripper_restart.close()
        await home.close()
        await closed_loop.close()
        await validation.close()
        await manager.close()
        await model_manager.close()

    app = FastAPI(
        title="ForceVLA Web", lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None
    )
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=config["allowed_hosts"])
    app.state.manager = manager
    app.state.control_validation = validation
    app.state.closed_loop = closed_loop
    app.state.home = home
    app.state.gripper_restart = gripper_restart
    app.state.model_manager = model_manager

    @app.exception_handler(Conflict)
    async def conflict_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(CheckFailed)
    async def validation_failure_handler(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.middleware("http")
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob:; connect-src 'self'; object-src 'none'; frame-ancestors 'none'"
        )
        return response

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/v1/session")
    def session(request: Request):
        try:
            owner = identity(request)
        except HTTPException:
            owner = secrets.token_hex(24)
        response = JSONResponse(
            {
                "csrf_token": sign("csrf:" + owner),
                "default_prompt": config.get("default_prompt", ""),
            }
        )
        response.set_cookie(
            "forcevla_session",
            owner + "." + sign(owner),
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
        )
        return response

    @app.get("/api/v1/status")
    def status(request: Request):
        value = manager.status(identity(request))
        value["model_manager"] = model_manager.public()
        return value

    @app.get("/api/v1/models")
    def models(request: Request):
        identity(request)
        return model_manager.public()

    @app.post("/api/v1/models/{model_id}/load", status_code=202)
    async def load_model(model_id: str, request: Request):
        command_owner(request)
        try:
            if model_manager.state in {"STARTING", "RUNNING"}:
                return model_manager.public()
            asyncio.create_task(model_manager.load(model_id))
            return model_manager.public()
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post("/api/v1/models/stop", status_code=202)
    async def stop_model(request: Request):
        command_owner(request)
        return await model_manager.stop()

    @app.get("/api/v1/gripper-restart/status")
    def gripper_restart_status(request: Request):
        return gripper_restart.status(identity(request))

    @app.post("/api/v1/gripper-restart/start", status_code=202)
    async def gripper_restart_start(options: HomeOptions, request: Request):
        return await gripper_restart.start(command_owner(request), options.command_id)

    @app.post("/api/v1/gripper-restart/check")
    async def gripper_restart_check(request: Request):
        owner = command_owner(request)
        try:
            return await gripper_restart.check(owner)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/v1/home/status")
    def home_status(request: Request):
        return home.status(identity(request))

    @app.post("/api/v1/home/start", status_code=202)
    async def home_start(options: HomeOptions, request: Request):
        return await home.start(command_owner(request), options.command_id)

    @app.post("/api/v1/home/check")
    async def home_check(request: Request):
        owner = command_owner(request)
        try:
            return await home.check(owner)
        except (OSError, RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get("/api/v1/control-validation/status")
    def validation_status(request: Request):
        return validation.status(identity(request))

    @app.post("/api/v1/control-validation/check")
    async def validation_check(request: Request):
        command_owner(request)
        return await validation.check()

    @app.post("/api/v1/control-validation/start", status_code=202)
    async def validation_start(options: ValidationOptions, request: Request):
        return await validation.start(command_owner(request), options.model_dump())

    @app.post("/api/v1/control-validation/abort", status_code=202)
    async def validation_abort(request: Request):
        return await validation.abort(command_owner(request))

    @app.post("/api/v1/control-validation/release")
    async def validation_release(options: ReleaseOptions, request: Request):
        return await validation.release(command_owner(request), options.safety_ack)

    @app.get("/api/v1/control-validation/report")
    def validation_report(request: Request):
        owner = identity(request)
        value = validation.status(owner)
        return JSONResponse(
            value, headers={"Content-Disposition": 'attachment; filename="control-validation.json"'}
        )

    @app.get("/api/v1/closed-loop/status")
    def closed_loop_status(request: Request):
        return closed_loop.status(identity(request))

    @app.post("/api/v1/closed-loop/runs", status_code=202)
    async def closed_loop_start(options: ClosedLoopOptions, request: Request):
        owner = command_owner(request)
        values = options.model_dump()
        values["prompt"] = values["prompt"].strip()
        if not values["prompt"]:
            raise HTTPException(422, "任务文本不能为空")
        return await closed_loop.start(owner, values)

    @app.get("/api/v1/closed-loop/runs")
    def closed_loop_runs(request: Request):
        return closed_loop.runs(identity(request))

    @app.get("/api/v1/closed-loop/runs/{run_id}")
    def closed_loop_run(run_id: str, request: Request):
        owner = identity(request)
        try:
            return closed_loop.read_run(run_id, owner)
        except (OSError, KeyError, ValueError):
            raise HTTPException(404, "闭环记录不存在") from None

    @app.post("/api/v1/closed-loop/runs/{run_id}/stop", status_code=202)
    async def closed_loop_stop(run_id: str, request: Request):
        owner = command_owner(request)
        try:
            return await closed_loop.stop(run_id, owner)
        except (OSError, KeyError, ValueError):
            raise HTTPException(404, "闭环记录不存在") from None

    @app.get("/api/v1/closed-loop/runs/{run_id}/export")
    def closed_loop_export(run_id: str, request: Request):
        owner = identity(request)
        try:
            output = closed_loop.export(run_id, owner)
        except (OSError, KeyError, ValueError):
            raise HTTPException(404, "闭环记录不存在") from None

        def chunks():
            try:
                while chunk := output.read(256 * 1024):
                    yield chunk
            finally:
                output.close()

        return StreamingResponse(
            chunks(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="closed-loop-{run_id}.zip"'},
        )

    @app.get("/api/v1/config")
    def get_config(request: Request):
        identity(request)
        return {
            "revision": manager.config_revision,
            "max_age_s": config["max_age_s"],
            "max_skew_s": config["max_skew_s"],
            "cameras": CAMERAS,
            "source": config["source"],
            "model": "forcevla-dual",
            "offline_ik": False,
        }

    @app.post("/api/v1/runs", status_code=202)
    async def start(options: RunOptions, request: Request):
        owner = command_owner(request)
        values = options.model_dump()
        values["prompt"] = values["prompt"].strip()
        if not values["prompt"]:
            raise HTTPException(422, "任务文本不能为空")
        try:
            return await manager.start(values, owner)
        except OSError as exc:
            raise HTTPException(507, str(exc)) from exc

    @app.post("/api/v1/runs/{run_id}/stop", status_code=202)
    async def stop(run_id: str, request: Request):
        return await manager.stop(run_id, command_owner(request))

    def require_run(run_id):
        run = manager.store.get_run(run_id)
        if run is None:
            raise HTTPException(404, "会话不存在")
        return run

    @app.get("/api/v1/runs")
    def runs(request: Request):
        identity(request)
        return [
            {
                k: v
                for k, v in run.items()
                if k not in ("owner", "config", "deployment_manifest", "web_source_sha256")
            }
            for run in manager.store.runs()
        ]

    @app.get("/api/v1/runs/{run_id}")
    def run(run_id: str, request: Request):
        identity(request)
        value = require_run(run_id)
        return {
            k: v
            for k, v in value.items()
            if k not in ("owner", "config", "deployment_manifest", "web_source_sha256")
        } | {"predictions": manager.store.predictions(run_id)}

    @app.get("/api/v1/runs/{run_id}/events")
    def events(run_id: str, request: Request, after_seq: int = 0):
        identity(request)
        require_run(run_id)
        return manager.store.events(run_id, max(0, after_seq))

    @app.get("/api/v1/predictions/{prediction_id}")
    def prediction(prediction_id: str, request: Request):
        identity(request)
        value = manager.store.get_prediction(prediction_id)
        if value is None:
            raise HTTPException(404, "预测不存在")
        return value

    @app.get("/api/v1/images/{camera}")
    def live_image(camera: str, request: Request):
        identity(request)
        snap = manager.snapshot
        if camera not in CAMERAS or snap is None or camera not in snap.images:
            raise HTTPException(404, "图像暂不可用")
        return Response(
            snap.jpeg(camera), media_type="image/jpeg", headers={"X-Snapshot-ID": snap.id}
        )

    @app.get("/api/v1/predictions/{prediction_id}/images/{camera}")
    def snapshot_image(prediction_id: str, camera: str, request: Request):
        identity(request)
        value = manager.store.get_prediction(prediction_id, with_snapshot=True)
        if camera not in CAMERAS or value is None:
            raise HTTPException(404, "输入快照不存在")
        return Response(Snapshot.unpack(value[1]).jpeg(camera), media_type="image/jpeg")

    @app.get("/api/v1/runs/{run_id}/export")
    def export(run_id: str, request: Request):
        identity(request)
        run = require_run(run_id)
        run.pop("owner", None)
        output = tempfile.TemporaryFile()
        try:
            with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(run, ensure_ascii=False, indent=2))
                with manager.store.connect() as db:
                    with archive.open("events.jsonl", "w") as events_file:
                        for seq, body in db.execute(
                            "SELECT seq,body FROM events WHERE run_id=? ORDER BY seq", (run_id,)
                        ):
                            event = {**json.loads(body), "event_seq": seq}
                            events_file.write(
                                (json.dumps(event, ensure_ascii=False) + "\n").encode()
                            )
                    for pid, body, blob in db.execute(
                        "SELECT id,body,snapshot FROM predictions WHERE run_id=?", (run_id,)
                    ):
                        archive.writestr(f"predictions/{pid}.json", body)
                        archive.writestr(f"snapshots/{pid}.npz", blob)
            output.seek(0)
        except Exception:
            output.close()
            raise

        def chunks():
            try:
                while chunk := output.read(256 * 1024):
                    yield chunk
            finally:
                output.close()

        return StreamingResponse(
            chunks(),
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="prediction-{run_id}.zip"'},
        )

    @app.post("/api/v1/connection-events")
    def connection_event(value: ConnectionCloseReport, request: Request):
        owner = command_owner(request)
        return {"recorded": manager.heartbeats.client_closed(owner, value.model_dump())}

    @app.websocket("/api/v1/events")
    async def websocket(ws: WebSocket):
        try:
            owner = identity(ws)
        except HTTPException:
            await ws.close(code=1008)
            return
        if not same_origin(ws):
            await ws.close(code=1008)
            return
        await ws.accept()
        try:
            connection_id = manager.heartbeats.open(
                owner, peer=str(ws.client), user_agent=ws.headers.get("user-agent", "")
            )
        except RuntimeError:
            await ws.close(code=1013)
            return
        manager.touch(owner)
        reason, code = "server_handler_finished", 1000
        async def receive_heartbeats():
            while True:
                message = await ws.receive_text()
                manager.heartbeats.receive(connection_id, message)

        async def send_status():
            while True:
                status = manager.status(owner)
                status["heartbeat"] = manager.heartbeats.challenge(connection_id)
                await asyncio.wait_for(ws.send_json(status), timeout=2)
                await asyncio.sleep(0.2)

        tasks = [asyncio.create_task(receive_heartbeats()), asyncio.create_task(send_status())]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect as exc:
            reason, code = "peer_disconnected:" + str(getattr(exc, "reason", "")), exc.code
        except TimeoutError:
            reason, code = "status_send_timeout", 1011
        except asyncio.CancelledError:
            reason, code = "server_handler_cancelled", 1001
            raise
        except RuntimeError as exc:
            reason, code = "websocket_runtime_error:" + str(exc), 1011
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            manager.heartbeats.close_connection(connection_id, reason=reason, code=code)
            try:
                await ws.close(code=code if code in (1000, 1001, 1011) else 1000)
            except (RuntimeError, WebSocketDisconnect):
                pass

    app.mount("/static", StaticFiles(directory=STATIC), name="static")
    return app
