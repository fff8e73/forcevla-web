import io
import json
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forcevla_web.app import create_app
from forcevla_web.manager import Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store


def wait_for(predicate, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.025)
    raise AssertionError("timed out")


@pytest.fixture
def loop_system(tmp_path):
    config = dict(
        source="demo",
        allowed_hosts=["testserver"],
        max_age_s=0.75,
        max_skew_s=0.2,
        max_source_age_s=1,
        disconnect_grace_s=10,
        data_dir=str(tmp_path),
        quota_mb=100,
        closed_loop=dict(enabled=True, integration_root="/home/user/forcevla-web/robot_control"),
    )
    manager = Manager(config, DemoSource(), DemoPolicy(), Store(tmp_path, 100))
    with TestClient(create_app(config, manager)) as client:
        session = client.get("/api/v1/session").json()
        client.headers.update(
            {"Origin": "http://testserver", "X-CSRF-Token": session["csrf_token"]}
        )
        yield client, manager


def start(client, **extra):
    return client.post(
        "/api/v1/closed-loop/runs",
        json={
            **dict(
                mode="simulate",
                prompt="synthetic task",
                seed=0,
                execution_horizon=3,
                max_cycles=3,
                max_task_duration_s=30,
                command_id="loop-test-1234",
            ),
            **extra,
        },
    )


def finished(client, run_id):
    value = client.get("/api/v1/closed-loop/runs/" + run_id).json()
    return (
        value
        if value.get("state")
        in ("COMPLETED_LIMIT", "FAILED", "TIMEOUT", "CANCELLED", "STOP_UNCONFIRMED")
        else None
    )


def test_three_real_worker_cycles_history_and_export(loop_system):
    client, manager = loop_system
    response = start(client)
    assert response.status_code == 202, response.text
    run_id = response.json()["id"]
    assert start(client).json()["id"] == run_id
    assert start(client, max_cycles=4).status_code == 409
    assert (
        client.post(
            "/api/v1/runs", json=dict(kind="single", prompt="x", command_id="prediction-other")
        ).status_code
        == 409
    )
    result = wait_for(lambda: finished(client, run_id))
    assert result["state"] == "COMPLETED_LIMIT", result
    assert result["completed_cycles"] == 3
    assert result["stop"]["confirmed"]
    assert result["synthetic"]
    assert client.get("/api/v1/closed-loop/runs").json()[0]["id"] == run_id
    exported = client.get("/api/v1/closed-loop/runs/" + run_id + "/export")
    with zipfile.ZipFile(io.BytesIO(exported.content)) as archive:
        assert "web.json" not in archive.namelist()
        states = []
        for cycle in (1, 2, 3):
            folder = f"cycles/{cycle:04d}/"
            record = json.loads(archive.read(folder + "cycle.json"))
            assert min(record["source_times"].values()) > record["hold_before"]["unix"]
            assert record["result"]["executed_model_steps"] == 3
            with np.load(
                io.BytesIO(archive.read(folder + "observation.npz")), allow_pickle=False
            ) as sample:
                states.append(sample["joints"].copy())
            with np.load(
                io.BytesIO(archive.read(folder + "plan/raw_prediction.npz")), allow_pickle=False
            ) as raw:
                assert raw["actions"].shape == (1, 50, 14)
        assert states[1][0] > states[0][0] and states[2][0] > states[1][0]


def test_cancel_permissions_and_block_live(loop_system):
    client, manager = loop_system
    assert start(client, mode="live").status_code == 409
    response = start(client, max_cycles=10)
    run_id = response.json()["id"]
    cookie = client.cookies.get("forcevla_session")
    client.cookies.clear()
    second = client.get("/api/v1/session").json()
    client.headers["X-CSRF-Token"] = second["csrf_token"]
    assert client.post("/api/v1/closed-loop/runs/" + run_id + "/stop").status_code == 409
    client.cookies.clear()
    client.cookies.set("forcevla_session", cookie)
    session = client.get("/api/v1/session").json()
    client.headers["X-CSRF-Token"] = session["csrf_token"]
    assert client.post("/api/v1/closed-loop/runs/" + run_id + "/stop").status_code == 202
    result = wait_for(lambda: finished(client, run_id))
    assert result["state"] == "CANCELLED", result
    assert result["stop"]["confirmed"]
    assert result["completed_cycles"] < 10


def test_invalid_input_and_origin(loop_system):
    client, _ = loop_system
    for options in (
        dict(seed=True),
        dict(execution_horizon=50),
        dict(max_cycles=0),
        dict(prompt="  "),
    ):
        assert start(client, **options).status_code == 422
    client.headers["Origin"] = "http://elsewhere"
    assert start(client).status_code == 403
    assert client.get("/api/v1/closed-loop/runs/missing").status_code == 404


@pytest.mark.parametrize(
    "fault,reason",
    [
        ("wrong_id", "model_response_id_mismatch"),
        ("nonfinite", "Invalid finite"),
        ("rpc_error", "injected_RPC_error"),
    ],
)
def test_late_or_invalid_model_never_executes(loop_system, fault, reason):
    client, manager = loop_system
    manager.closed_loop.options["simulation_fault"] = fault
    run = start(client).json()
    result = wait_for(lambda: finished(client, run["id"]))
    assert result["state"] == "FAILED", result
    assert reason in result["reason"]
    assert result["completed_cycles"] == 0
    assert result["execution"]["current_segment_ticks"] == 0
    assert result["stop"]["confirmed"]


def test_old_observation_is_not_replayed_after_first_cycle(loop_system):
    client, manager = loop_system
    manager.closed_loop.options["simulation_freeze_after"] = 2
    run = start(client).json()

    def poll():
        manager.touch(manager.closed_loop.current["owner"])
        return finished(client, run["id"])

    result = wait_for(poll)
    assert result["state"] == "FAILED", result
    assert result["completed_cycles"] == 1, result
    assert "observation_timeout" in result["reason"]
    assert result["stop"]["confirmed"]


def test_timeout_during_inference_keeps_hold_and_discards_result(loop_system):
    client, manager = loop_system
    manager.closed_loop.options["simulation_rpc_s"] = 2.3
    run = start(client).json()

    def poll():
        manager.touch(manager.closed_loop.current["owner"])
        return finished(client, run["id"])

    result = wait_for(poll)
    assert result["state"] == "FAILED", result
    assert result["reason"] == "infer_timeout"
    assert result["execution"]["current_segment_ticks"] == 0
    assert result["stop"]["confirmed"]


def test_missing_process_requires_review_instead_of_resuming(loop_system):
    client, manager = loop_system
    root = manager.closed_loop.root
    run_id = "a" * 32
    folder = root / run_id
    folder.mkdir()
    (folder / "web.json").write_text(
        json.dumps(dict(owner="past", created_unix=time.time(), options={"mode": "live"}))
    )
    (folder / "status.json").write_text(json.dumps(dict(id=run_id, state="EXECUTE", mode="live")))
    manager.closed_loop.current = dict(id=run_id, pid=999999999, process_signature="old")
    status = client.get("/api/v1/closed-loop/status").json()
    assert status["run"]["state"] == "INTERRUPTED_REVIEW"
    assert status["blocked"]
    assert start(client).status_code == 409


@pytest.mark.parametrize("phase", ["PREPARE", "OBSERVE"])
def test_cancel_preparation_and_observation_never_submits_motion(loop_system, phase):
    client, manager = loop_system
    if phase == "OBSERVE":
        manager.closed_loop.options["simulation_freeze_after"] = 1
    run = start(client, max_cycles=10).json()

    def in_phase():
        manager.touch(manager.closed_loop.current["owner"])
        value = client.get("/api/v1/closed-loop/runs/" + run["id"]).json()
        return value if value["state"] == phase else None

    wait_for(in_phase)
    assert client.post("/api/v1/closed-loop/runs/" + run["id"] + "/stop").status_code == 202
    result = wait_for(lambda: finished(client, run["id"]))
    assert result["state"] == "CANCELLED", result
    assert result["execution"]["current_segment_ticks"] == 0
    assert result["stop"]["confirmed"]


def test_task_deadline_is_distinct_from_model_error(loop_system):
    client, manager = loop_system
    manager.closed_loop.options["simulation_rpc_s"] = 0.4
    run = start(client, max_task_duration_s=0.3).json()
    result = wait_for(lambda: finished(client, run["id"]))
    assert result["state"] == "TIMEOUT", result
    assert result["reason"] == "task_timeout"
    assert result["completed_cycles"] == 0
    assert result["stop"]["confirmed"]


def test_browser_lease_expires_without_resuming_task(loop_system):
    client, manager = loop_system
    manager.closed_loop.profile["browser_disconnect_timeout_s"] = 0.2
    manager.closed_loop.options["simulation_rpc_s"] = 0.4
    run = start(client).json()
    result = wait_for(lambda: finished(client, run["id"]))
    assert result["state"] == "CANCELLED", result
    assert result["reason"] == "browser_disconnected"
    assert result["completed_cycles"] == 0
    assert result["stop"]["confirmed"]


def test_movej_api_contract_and_demo_cannot_move(loop_system):
    client, manager = loop_system
    for options in (dict(engine='movej',mode='live',execution_horizon=51),
                    dict(engine='movej',mode='simulate',execution_horizon=20),
                    dict(engine='native',mode='live',execution_horizon=20)):
        assert start(client, **options).status_code == 422
    manager.closed_loop.options['planner_enabled'] = True
    assert start(client, engine='movej',mode='live',execution_horizon=20).status_code == 409


@pytest.mark.parametrize("steps, cycles", [(1, 50), (7, 3), (20, 50), (50, 100)])
def test_movej_dispatch_uses_configured_prefix_worker(loop_system, monkeypatch, steps, cycles):
    client, manager = loop_system
    manager.closed_loop.config['source'] = 'live'
    manager.closed_loop.options['planner_enabled'] = True
    manager.healthy = True
    manager.input = {'ready': True}
    calls=[]
    async def capture(*args, **kwargs):
        calls.append(args)
        raise OSError('test intercept: no hardware process started')
    monkeypatch.setattr('forcevla_web.closed_loop.asyncio.create_subprocess_exec', capture)
    response=start(client,engine='movej',mode='live',execution_horizon=steps,max_cycles=cycles,max_task_duration_s=120)
    assert response.status_code == 409
    assert calls[0][1] == 'planner-loop'
    root=manager.closed_loop.root / manager.closed_loop.current['id']
    profile=json.loads((root/'profile.json').read_text())
    assert profile['execution_horizon']==steps
    assert profile['max_cycles']==cycles
    assert 'max_joint_speed_rad_s' not in profile
    assert 'max_joint_acceleration_rad_s2' not in profile
    assert client.get('/api/v1/closed-loop/status').json()['blocked'] is False


def test_movej_defaults_and_parameter_bounds():
    from pydantic import ValidationError
    from forcevla_web.app import ClosedLoopOptions
    base = dict(engine="movej", mode="live", prompt="test", command_id="parameters-test")
    options = ClosedLoopOptions(**base)
    assert options.execution_horizon == 20
    assert options.max_cycles == 50
    assert options.max_task_duration_s == 600
    for bad in (dict(execution_horizon=0), dict(execution_horizon=51),
                dict(execution_horizon=1.5), dict(execution_horizon=True),
                dict(max_cycles=0), dict(max_cycles=101), dict(max_cycles=True)):
        with pytest.raises(ValidationError):
            ClosedLoopOptions(**base, **bad)
    native = ClosedLoopOptions(mode="simulate", prompt="test", command_id="native-defaults")
    assert native.execution_horizon == 3
