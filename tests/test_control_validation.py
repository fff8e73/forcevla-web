import threading
import time

import pytest
from fastapi.testclient import TestClient

from forcevla_web.app import create_app
from forcevla_web.control_validation import telemetry_report
from forcevla_web.manager import Manager
from forcevla_web.observation import DemoSource
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store


class Source(DemoSource):
    moving = False
    stale = False

    def read(self):
        snap = super().read()
        now = time.monotonic() - (10 if self.stale else 0)
        snap.meta["control_feedback"] = {
            "joints": {
                "positions": [0.1] * 14,
                "velocities": [0.1 if self.moving else 0.0] * 14,
                "received_mono": now,
                "source_time": time.time(),
            },
            "left_gripper": {"position_rad": -0.034, "velocity_rad_s": 0.0, "received_mono": now},
            "right_gripper": {"position_rad": -0.034, "velocity_rad_s": 0.0, "received_mono": now},
        }
        return snap


class ApexFake:
    def __init__(self):
        self.posts = []
        self.mode = 1
        self.boot = "boot"
        self.session = None
        self.reserved = False
        self.state = "idle"
        self.fail_claim = False
        self.fail_hold = False
        self.release_uncertain = False
        self.claimed_at = None
        self.stopped = False
        self.slow_claim = False
        self.entered = threading.Event()
        self.continue_claim = threading.Event()
        self.continue_claim.set()

    def get(self, name):
        if name == "robot":
            return {
                "control_ready": True,
                "control_error": False,
                "arm_state_age_sec": 0.01,
                "current_input_mode": self.mode,
                "current_mode": 3,
            }
        if name == "replay":
            return {"is_playing": False, "is_paused": False}
        if self.claimed_at and time.monotonic() - self.claimed_at > 2.5 and self.reserved:
            self.state = "held"
        return {
            "available": True,
            "healthy": True,
            "reserved": self.reserved,
            "boot_id": self.boot,
            "session_id": self.session,
            "state": self.state,
            "reason": "网页心跳超时" if self.state == "held" else "",
            "status_age_sec": 0.01,
            "actual": [-0.034 / 1.6] * 2,
            "applied_target": [0.0, 0.0],
            "calibration": [[0.0, 1.6], [0.0, 1.6]],
            "motor_feedback_age": [0.001, 0.001],
            "issued": [0.0, 0.0],
        }

    def select_none(self):
        self.posts.append("input_none")
        self.mode = 0
        return {"success": True}

    def gripper(self, action, boot, session, safety_ack=False):
        self.posts.append(action)
        assert action in {"claim", "hold", "release"}
        assert boot == self.boot
        if action == "claim":
            self.entered.set()
            if self.slow_claim:
                assert self.continue_claim.wait(3)
            if self.fail_claim:
                raise TimeoutError("claim ack lost")
            assert safety_ack and not self.reserved
            self.session = session
            self.reserved = True
            self.state = "active"
            self.claimed_at = time.monotonic()
        elif action == "hold":
            if self.fail_hold:
                raise TimeoutError("hold ack lost")
            assert self.session == session
            self.state = "held"
        else:
            assert safety_ack and self.session == session
            if self.release_uncertain:
                raise TimeoutError("release ack lost")
            self.session = None
            self.reserved = False
            self.state = "idle"
        return self.get("gripper")


def wait_for(predicate, timeout=5):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        value = predicate()
        if value:
            return value
        time.sleep(0.02)
    raise AssertionError("timeout")


@pytest.fixture
def system(tmp_path):
    config = {
        "source": "live",
        "allowed_hosts": ["testserver"],
        "max_age_s": 0.75,
        "max_skew_s": 0.2,
        "max_source_age_s": 1,
        "disconnect_grace_s": 10,
        "data_dir": str(tmp_path),
        "quota_mb": 100,
        "control_validation": {"enabled": True, "poll_period_s": 0.02},
    }
    source = Source()
    backend = ApexFake()
    manager = Manager(config, source, DemoPolicy(), Store(tmp_path, 100))
    app = create_app(config, manager, validation_backend=backend)
    with TestClient(app) as client:
        s = client.get("/api/v1/session").json()
        client.headers.update({"Origin": "http://testserver", "X-CSRF-Token": s["csrf_token"]})
        wait_for(lambda: manager.snapshot is not None)
        yield client, manager, backend, source, app.state.control_validation
        backend.continue_claim.set()


def start(client, command="validation-request-1", kind="stationary_cycle", ack=True):
    return client.post(
        "/api/v1/control-validation/start",
        json={"kind": kind, "command_id": command, "safety_ack": ack},
    )


def done(client):
    result = client.get("/api/v1/control-validation/status").json()
    run = result["run"]
    return (
        run
        if run
        and run["state"]
        not in {
            "CHECKING",
            "SELECTING_NONE",
            "CLAIMING",
            "OBSERVING",
            "HOLDING",
            "RELEASING",
            "ABORTING",
        }
        else None
    )


def test_check_is_readonly_and_negative_feedback_allowed(system):
    client, manager, backend, source, validation = system
    result = client.post("/api/v1/control-validation/check").json()
    assert result["ready"]
    assert backend.posts == []
    assert result["actual_gripper"][0] < 0


def test_explicit_ack_origin_required(system):
    client, _, backend, _, _ = system
    assert start(client, ack=False).status_code == 409
    assert start(client, ack="true").status_code == 422
    assert (
        client.post(
            "/api/v1/control-validation/check", headers={"Origin": "http://evil"}
        ).status_code
        == 403
    )
    assert backend.posts == []


def test_stationary_cycle_only_allowed_operations(system):
    client, _, backend, _, _ = system
    assert start(client).status_code == 202
    run = wait_for(lambda: done(client))
    assert run["state"] == "COMPLETED_STATIONARY", run
    assert backend.posts == ["input_none", "claim", "hold", "release"]
    assert backend.mode == 0 and not backend.reserved
    assert run["moving_stop_verified"] is False and run["target_commands_sent"] == 0
    assert "driver_session" not in run
    assert start(client).status_code == 202
    assert backend.posts == ["input_none", "claim", "hold", "release"]
    assert client.get("/api/v1/control-validation/report").status_code == 200


def test_moving_feedback_rejected_before_post(system):
    client, manager, backend, source, _ = system
    source.moving = True
    wait_for(lambda: manager.snapshot.meta["control_feedback"]["joints"]["velocities"][0] > 0.02)
    start(client)
    run = wait_for(lambda: done(client))
    assert run["state"] == "BLOCKED"
    assert backend.posts == []


def test_stale_feedback_rejected_before_post(system):
    client, manager, backend, source, _ = system
    source.stale = True
    time.sleep(0.2)
    start(client)
    run = wait_for(lambda: done(client))
    assert run["state"] == "BLOCKED"
    assert backend.posts == []


def test_other_owner_and_inference_mutually_excluded(system):
    client, manager, backend, source, validation = system
    backend.slow_claim = True
    backend.continue_claim.clear()
    start(client)
    assert backend.entered.wait(2)
    assert (
        client.post(
            "/api/v1/runs",
            json={
                "kind": "single",
                "prompt": "test",
                "seed": 0,
                "command_id": "infer-after-control",
            },
        ).status_code
        == 409
    )
    saved = client.cookies.get("forcevla_session")
    csrf = client.headers["X-CSRF-Token"]
    client.cookies.clear()
    s = client.get("/api/v1/session").json()
    client.headers["X-CSRF-Token"] = s["csrf_token"]
    assert client.post("/api/v1/control-validation/abort").status_code == 409
    assert start(client).status_code == 409
    client.cookies.clear()
    client.cookies.set("forcevla_session", saved)
    client.headers["X-CSRF-Token"] = csrf
    client.post("/api/v1/control-validation/abort")
    backend.continue_claim.set()
    run = wait_for(lambda: done(client))
    assert run["state"] == "HELD_REVIEW"
    assert backend.reserved
    assert "release" not in backend.posts
    assert (
        client.post("/api/v1/control-validation/release", json={"safety_ack": True}).status_code
        == 200
    )
    assert not backend.reserved


def test_hold_failure_never_claims_stopped(system):
    client, _, backend, _, _ = system
    backend.fail_hold = True
    start(client)
    run = wait_for(lambda: done(client))
    assert run["state"] == "UNCERTAIN"
    assert backend.reserved
    assert "release" not in backend.posts
    assert not run.get("hold_acknowledged", False)


def test_lost_claim_ack_no_automatic_retry(system):
    client, _, backend, _, _ = system
    backend.fail_claim = True
    start(client)
    run = wait_for(lambda: done(client))
    assert run["state"] == "ABORTED_UNOWNED"
    assert backend.posts.count("claim") == 1 and backend.posts.count("release") == 0


def test_lease_expiry_observed_not_moving_stop(system):
    client, _, backend, _, _ = system
    start(client, kind="lease_expiry")
    run = wait_for(lambda: done(client), timeout=6)
    assert run["state"] == "COMPLETED_STATIONARY", run
    assert run["lease_hold_observed"] is True
    assert not run["moving_stop_verified"]


def test_demo_never_reaches_apex(system):
    client, _, backend, _, validation = system
    validation.enabled = False
    assert start(client).status_code == 409
    assert client.post("/api/v1/control-validation/check").status_code == 409
    assert backend.posts == []


def test_missing_velocity_is_not_assumed_zero():
    source = Source()
    snapshot = source.read()
    snapshot.meta["control_feedback"]["left_gripper"].pop("velocity_rad_s")
    backend = ApexFake()
    result = telemetry_report(
        snapshot, backend.get("robot"), backend.get("gripper"), backend.get("replay")
    )
    assert not result["ready"]


def test_restart_requires_review_and_does_not_resume(system):
    from forcevla_web.control_validation import ControlValidation

    client, manager, backend, source, validation = system
    validation.current = {
        "id": "a" * 32,
        "owner": "owner",
        "command_id": "restart-test",
        "state": "OBSERVING",
        "events": [],
        "driver_session": "session",
        "driver_boot": "boot",
        "target_commands_sent": 0,
    }
    validation.save()
    restored = ControlValidation(manager, validation.config, backend)
    assert restored.current["state"] == "INTERRUPTED_REVIEW"
    assert restored.blocked and restored.job is None and backend.posts == []
    validation.current = None


def test_lost_ack_after_claim_is_held_without_release(system):
    client, _, backend, _, _ = system
    original = backend.gripper

    def lost_ack(action, *args, **kwargs):
        result = original(action, *args, **kwargs)
        if action == "claim":
            raise TimeoutError("ack lost after real claim")
        return result

    backend.gripper = lost_ack
    start(client)
    run = wait_for(lambda: done(client))
    assert run["state"] == "HELD_REVIEW"
    assert backend.posts == ["input_none", "claim", "hold"]
    assert backend.reserved and backend.state == "held"


def test_disconnect_owner_holds_without_release(system):
    client, manager, backend, _, validation = system
    backend.slow_claim = True
    backend.continue_claim.clear()
    start(client)
    assert backend.entered.wait(2)
    manager.last_seen[validation.current["owner"]] = 0
    backend.continue_claim.set()
    run = wait_for(lambda: done(client))
    assert run["state"] == "HELD_REVIEW" and "断开" in run["reason"]
    assert backend.reserved and "release" not in backend.posts


@pytest.mark.parametrize(
    "velocity,ready",
    [(0.021978, True), (-0.021978, True), (0.03, True), (0.0301, False), (-0.0301, False)],
)
def test_web_gripper_speed_margin(velocity, ready):
    snapshot = Source().read()
    for name in ["left_gripper", "right_gripper"]:
        snapshot.meta["control_feedback"][name]["velocity_rad_s"] = velocity
    backend = ApexFake()
    result = telemetry_report(
        snapshot,
        backend.get("robot"),
        backend.get("gripper"),
        backend.get("replay"),
        gripper_speed_limit=0.03,
    )
    assert result["ready"] is ready
    assert result["web_gripper_speed_limit_rad_s"] == 0.03
    assert result["driver_claim_speed_limit_rad_s"] == 0.02


def test_gripper_margin_does_not_relax_arm_or_position_checks():
    snapshot = Source().read()
    backend = ApexFake()
    snapshot.meta["control_feedback"]["joints"]["velocities"][0] = 0.025
    result = telemetry_report(
        snapshot, backend.get("robot"), backend.get("gripper"), backend.get("replay")
    )
    assert not result["ready"] and any("关节速度" in r for r in result["reasons"])
    snapshot = Source().read()
    snapshot.meta["control_feedback"]["left_gripper"]["position_rad"] = 0.06
    result = telemetry_report(
        snapshot, backend.get("robot"), backend.get("gripper"), backend.get("replay")
    )
    assert not result["ready"] and any("原目标偏差" in r for r in result["reasons"])
