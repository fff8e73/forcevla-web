import io
import json
import threading
import time
import zipfile

import numpy as np
import pytest
from fastapi.testclient import TestClient

from forcevla_web.app import create_app
from forcevla_web.manager import Manager
from forcevla_web.observation import DemoSource, Snapshot
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store


class ControlledPolicy(DemoPolicy):
    def __init__(self):
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = 0
        self.fail = False

    def predict(self, raw, request_id):
        self.calls += 1
        self.entered.set()
        assert self.release.wait(3)
        if self.fail:
            raise TimeoutError("injected RPC timeout")
        return super().predict(raw, request_id)


def wait_for(predicate, timeout=4):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        result = predicate()
        if result:
            return result
        time.sleep(0.02)
    raise AssertionError("condition timed out")


@pytest.fixture
def system(tmp_path):
    config = {
        "source": "demo",
        "allowed_hosts": ["testserver"],
        "max_age_s": 0.75,
        "max_skew_s": 0.2,
        "max_source_age_s": 1,
        "disconnect_grace_s": 10,
        "data_dir": str(tmp_path),
        "quota_mb": 100,
    }
    source, policy, store = DemoSource(), ControlledPolicy(), Store(tmp_path, 100)
    manager = Manager(config, source, policy, store)
    with TestClient(create_app(config, manager)) as client:
        session = client.get("/api/v1/session").json()
        client.headers.update(
            {"Origin": "http://testserver", "X-CSRF-Token": session["csrf_token"]}
        )
        wait_for(lambda: client.get("/api/v1/status").json()["input"]["ready"])
        yield client, manager, policy
        policy.release.set()


def start(client, **kw):
    return client.post(
        "/api/v1/runs",
        json={
            "kind": "single",
            "prompt": "test task",
            "seed": 42,
            "command_id": "request-123456",
            **kw,
        },
    )


def test_single_snapshot_export_and_readonly_routes(system):
    client, manager, policy = system
    response = start(client)
    assert response.status_code == 202
    run_id = response.json()["id"]
    wait_for(lambda: manager.run["state"] == "IDLE")
    assert manager.latest["actions"] and len(manager.latest["actions"]) == 50
    assert manager.latest["offline_ik"] == "unavailable"
    pid = manager.latest["id"]
    assert (
        client.get(f"/api/v1/predictions/{pid}/images/image").headers["content-type"]
        == "image/jpeg"
    )
    export = client.get(f"/api/v1/runs/{run_id}/export")
    with zipfile.ZipFile(io.BytesIO(export.content)) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        assert "owner" not in manifest
        saved = Snapshot.unpack(archive.read(f"snapshots/{pid}.npz"))
        np.testing.assert_allclose(saved.state, manager.latest["state"])
        assert json.loads(archive.read(f"predictions/{pid}.json"))["snapshot_id"] == saved.id
    assert not client.get("/api/v1/status").json()["capabilities"]["hardware_write"]
    assert client.post("/api/v1/control").status_code == 404
    assert client.get("/../config/local.json").status_code == 404
    assert client.get("/api/v1/predictions/nonexistent").status_code == 404


def test_stop_discards_inflight_and_duplicate_start(system):
    client, manager, policy = system
    policy.release.clear()
    run = start(client, kind="continuous").json()
    assert policy.entered.wait(2)
    assert start(client, kind="continuous").json()["id"] == run["id"]
    assert start(client, command_id="different-command").status_code == 409
    assert client.post(f"/api/v1/runs/{run['id']}/stop").status_code == 202
    assert manager.run["state"] == "STOPPING"
    assert start(client, command_id="different-command").status_code == 409
    policy.release.set()
    wait_for(lambda: manager.run["state"] == "IDLE")
    assert manager.latest is None
    assert policy.calls == 1
    predictions = manager.store.predictions(run["id"])
    assert len(predictions) == 1 and predictions[0]["discarded"]
    assert client.post(f"/api/v1/runs/{run['id']}/stop").status_code == 202


def test_session_origin_and_ownership(system):
    client, manager, policy = system
    assert start(client, prompt="  ").status_code == 422
    assert start(client, seed=True).status_code == 422
    assert (
        client.post(
            "/api/v1/runs",
            headers={"Origin": "http://evil.test"},
            json={"kind": "single", "prompt": "x", "command_id": "abcdefgh"},
        ).status_code
        == 403
    )
    policy.release.clear()
    run = start(client).json()
    old_cookie = client.cookies.get("forcevla_session")
    client.cookies.clear()
    second = client.get("/api/v1/session").json()
    client.headers["X-CSRF-Token"] = second["csrf_token"]
    assert client.post(f"/api/v1/runs/{run['id']}/stop").status_code == 409
    client.cookies.clear()
    client.cookies.set("forcevla_session", old_cookie)
    session = client.get("/api/v1/session").json()
    client.headers["X-CSRF-Token"] = session["csrf_token"]
    client.post(f"/api/v1/runs/{run['id']}/stop")
    policy.release.set()


def test_timeout_does_not_retry(system):
    client, manager, policy = system
    policy.fail = True
    assert start(client, kind="continuous").status_code == 202
    wait_for(lambda: manager.run["state"] == "ERROR")
    time.sleep(0.25)
    assert policy.calls == 1
    assert "timeout" in manager.run["reason"]


def test_disconnect_stops_without_client_command(system):
    client, manager, policy = system
    manager.config["disconnect_grace_s"] = 0.15
    policy.release.clear()
    start(client, kind="continuous")
    assert policy.entered.wait(2)
    wait_for(lambda: manager.run["state"] == "STOPPING")
    policy.release.set()
    wait_for(lambda: manager.run["state"] == "IDLE")
    assert manager.run["discarded"] == 1


def test_quota_blocks_new_run(system):
    client, manager, policy = system
    manager.store.quota = 100
    assert start(client).status_code == 507
    assert policy.calls == 0


def test_input_contract_faults_and_snapshot_roundtrip():
    config = {"max_age_s": 0.75, "max_skew_s": 0.2, "max_source_age_s": 1}
    s = DemoSource().read()
    assert s.validate(config)["ready"]
    restored = Snapshot.unpack(s.pack())
    np.testing.assert_array_equal(restored.images["image"], s.images["image"])
    assert not s.validate(config, now=time.monotonic() + 2)["ready"]
    s.meta["sources"]["image"]["source_time"] -= 3
    assert any("源采样" in r for r in s.validate(config)["reasons"])
    s = DemoSource().read()
    s.meta["motor_feedback_age"] = [5, 0]
    assert not s.validate(config)["ready"]
    s = DemoSource().read()
    s.state[2] = np.nan
    assert not s.validate(config)["ready"]
    s = DemoSource().read()
    del s.images["wrist_image"]
    assert not s.validate(config)["ready"]
    s = DemoSource().read()
    s.meta["sources"]["image"]["received_mono"] -= 0.3
    assert any("偏差" in r for r in s.validate(config)["reasons"])


def test_websocket_requires_origin_and_delivers_status(system):
    client, manager, policy = system
    with client.websocket_connect("/api/v1/events", headers={"origin": "http://testserver"}) as ws:
        ws.send_text("ping")
        data = ws.receive_json()
        assert data["mode"] == "prediction"
        assert data["input"]["ready"]


def test_nonfinite_source_time_fails_closed():
    config = {"max_age_s": 0.75, "max_skew_s": 0.2, "max_source_age_s": 1}
    snapshot = DemoSource().read()
    snapshot.meta["sources"]["gripper"]["received_mono"] = float("nan")
    result = snapshot.validate(config)
    assert not result["ready"]
    json.dumps(result, allow_nan=False)


def test_feedback_alignment_uses_image_time_not_latest_state():
    from forcevla_web.observation import align_feedback

    old = (np.array([0.1, 0.2]), {"received_mono": 1.0, "source_time": 10.0}, [0.02, 0.03])
    new = (np.array([0.9, 0.8]), {"received_mono": 1.4, "source_time": 10.4}, [0.01, 0.01])
    values, sources, ages = align_feedback(
        {"gripper": [old, new]}, {"received_mono": 1.02, "source_time": 10.02}
    )
    np.testing.assert_array_equal(values["gripper"], old[0])
    assert sources["gripper"]["received_mono"] == 1.0
    assert ages == [0.02, 0.03]


def test_store_connections_close_after_transaction(tmp_path):
    import sqlite3

    store = Store(tmp_path, 100)
    with store.connect() as db:
        db.execute("SELECT 1")
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        db.execute("SELECT 1")


def test_camera_transport_distinguishes_host_and_process(monkeypatch):
    from types import SimpleNamespace

    from forcevla_web import transport

    monkeypatch.setattr(transport, "udp_drops", lambda: 7)
    pub = SimpleNamespace(endpoint_gid=bytes.fromhex("010f0011000000010000000100000000"))
    reader = SimpleNamespace(
        node_name="observer", endpoint_gid=bytes.fromhex("010f0011000000020000000200000000")
    )
    node = SimpleNamespace(
        get_name=lambda: "observer",
        get_publishers_info_by_topic=lambda _: [pub],
        get_subscriptions_info_by_topic=lambda _: [reader],
    )
    assert transport.camera_transport(node, "/image")["same_host_prefix"] is True
    pub.endpoint_gid = bytes.fromhex("010f0022000000010000000100000000")
    report = transport.camera_transport(node, "/image")
    assert report["same_host_prefix"] is False and report["warning"]
    assert report["udp_drops_total"] == 7


def test_session_survives_web_app_recreation(system):
    client, manager, policy = system
    cookie = client.cookies.get("forcevla_session")
    config = manager.config
    other = Manager(config, DemoSource(), DemoPolicy(), manager.store)
    with TestClient(create_app(config, other)) as restarted:
        restarted.cookies.set("forcevla_session", cookie)
        assert restarted.get("/api/v1/status").status_code == 200
        assert (
            restarted.get("/api/v1/session").json()["csrf_token"] == client.headers["X-CSRF-Token"]
        )
    assert (manager.store.root / "session.key").stat().st_mode & 0o777 == 0o600


def test_server_heartbeat_response_and_close_event(system):
    client, manager, _ = system
    with client.websocket_connect('/api/v1/events', headers={'origin': 'http://testserver'}) as ws:
        status = ws.receive_json()
        heartbeat = status['heartbeat']
        owner = next(iter(manager.last_seen))
        ws.send_json(dict(kind='heartbeat_ack', connection_id=heartbeat['connection_id'],
                          sequence=heartbeat['sequence'], nonce=heartbeat['nonce'],
                          visibility_state='hidden', has_focus=False))
        wait_for(lambda: manager.heartbeats.summary(owner)['protocol'] == 'server_challenge_v1')
        assert client.get('/api/v1/status').json()['connection_health']['visibility_state'] == 'hidden'
    health = client.get('/api/v1/status').json()['connection_health']
    assert health['state'] == 'closed'
    report = dict(connection_id=heartbeat['connection_id'], code=1000,
                  reason='test close', was_clean=True, visibility_state='hidden', has_focus=False)
    assert client.post('/api/v1/connection-events', json=report).json()['recorded']
    assert client.post('/api/v1/connection-events', json=report,
                       headers={'Origin': 'http://elsewhere'}).status_code == 403
