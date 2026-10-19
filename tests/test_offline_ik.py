import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from forcevla_web.observation import JOINT_NAMES, DemoSource, ordered_joints
from forcevla_web.offline_ik import OfflineChecks, seed_request

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = json.loads((ROOT / "tests/fixtures/joint_fk_reference.json").read_text())


@pytest.fixture(scope="module")
def native():
    worker = ROOT / "native/build/offline_ik_worker"
    if not worker.exists():
        pytest.skip("Run scripts/build_ik.sh to build the isolated worker")
    p = subprocess.Popen(
        [str(worker), str(ROOT / "config/ik.json")],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    hello = json.loads(p.stdout.readline())
    assert hello["ready"], hello

    def call(**payload):
        p.stdin.write(json.dumps({"request_id": "test", **payload}) + "\n")
        p.stdin.flush()
        return json.loads(p.stdout.readline())

    yield call, hello["metadata"]
    p.stdin.close()
    p.wait(timeout=3)


def actions_for(inspect):
    result = []
    for side in ["left", "right"]:
        pose = inspect["tcp"][side]
        result.extend(
            pose["position"]
            + Rotation.from_quat(pose["quaternion_xyzw"]).as_euler("xyz").tolist()
            + [0.5]
        )
    return result


def test_fk_matches_frozen_controller_feedback(native):
    call, meta = native
    result = call(operation="inspect", seed=REFERENCE["joints"]["positions"])
    assert meta["joint_names"] == list(JOINT_NAMES)
    assert meta["collision_pair_count"] == 8
    for side in ["left", "right"]:
        np.testing.assert_allclose(
            result["tcp"][side]["position"], REFERENCE[side]["position"], atol=1e-6
        )
        a = Rotation.from_quat(result["tcp"][side]["quaternion_xyzw"])
        b = Rotation.from_quat(REFERENCE[side]["quaternion"])
        assert (a.inv() * b).magnitude() < 1e-6


def test_hold_and_small_path(native):
    call, meta = native
    seed = REFERENCE["joints"]["positions"]
    action = actions_for(call(operation="inspect", seed=seed))
    hold = call(seed=seed, actions=[action] * 50)
    assert hold["passed"] and hold["checked_steps"] == 50
    assert not hold["execution_verified"]
    path = np.tile(action, (50, 1))
    path[:, 0] += np.linspace(0, 0.01, 50)
    path[:, 7] += np.linspace(0, 0.01, 50)
    result = call(seed=seed, actions=path.tolist())
    assert result["passed"], result["reasons"]
    assert all(
        r["interpolation_samples"] == r["interpolation_samples_required"] for r in result["steps"]
    )


def test_invalid_seed_and_mismatched_tcp_are_rejected(native):
    call, _ = native
    seed = REFERENCE["joints"]["positions"]
    action = actions_for(call(operation="inspect", seed=seed))
    bad = seed.copy()
    bad[1] = 0.2
    assert call(seed=bad, actions=[action])["status"] == "invalid_seed"
    tcp = [action[:6], action[7:13]]
    tcp[0][0] += 0.1
    result = call(seed=seed, actions=[action], observed_tcp=tcp)
    assert result["status"] == "invalid_seed" and "seed_tcp_mismatch" in result["reasons"]


def test_unreachable_gripper_and_velocity(native):
    call, _ = native
    seed = REFERENCE["joints"]["positions"]
    action = actions_for(call(operation="inspect", seed=seed))
    bad = action.copy()
    bad[0] = 10
    result = call(seed=seed, actions=[bad])
    assert not result["passed"] and "ik_not_converged" in result["reasons"]
    bad = action.copy()
    bad[6] = 1.1
    assert "gripper_range" in call(seed=seed, actions=[bad])["reasons"]
    other = seed.copy()
    other[0] += 0.6
    far = actions_for(call(operation="inspect", seed=other))
    result = call(seed=seed, actions=[far])
    assert not result["passed"] and "joint_velocity" in result["reasons"]


def test_configured_collision_is_detected(native):
    call, meta = native
    rng = np.random.default_rng(9)
    for _ in range(100):
        seed = rng.uniform(meta["joint_lower"], meta["joint_upper"]).tolist()
        pose = call(operation="inspect", seed=seed)
        if pose["collision"]["distance_m"] < 0.005:
            report = call(seed=seed, actions=[actions_for(pose)])
            assert not report["passed"]
            assert "configured_collision_clearance" in report["reasons"]
            return
    raise AssertionError("No configured collision fixture found")


def test_name_order_and_seed_time_contract():
    names = list(reversed(JOINT_NAMES))
    values = list(range(14))
    np.testing.assert_equal(ordered_joints(names, values), list(reversed(values)))
    with pytest.raises(ValueError):
        ordered_joints(names[:-1], values[:-1])
    with pytest.raises(ValueError):
        ordered_joints([names[0]] * 14, values)
    snapshot = DemoSource().read()
    source = snapshot.meta["sources"]["image"].copy()
    snapshot.meta["joints"] = {
        "names": list(JOINT_NAMES),
        "positions": REFERENCE["joints"]["positions"],
        "source": source,
    }
    config = {"max_age_s": 0.75, "max_skew_s": 0.2, "max_source_age_s": 1.0}
    data, context = seed_request(snapshot, config)
    assert data["seed"] == REFERENCE["joints"]["positions"]
    # Analysis may occur much later: freshness is checked at the recorded snapshot.
    snapshot.meta["joints"]["source"]["received_mono"] -= 2
    with pytest.raises(ValueError, match="过期"):
        seed_request(snapshot, config)
    del snapshot.meta["joints"]
    with pytest.raises(ValueError):
        seed_request(snapshot, config)


def test_queue_bounded_and_late_result_never_replaces_new_prediction():
    class Store:
        def __init__(self):
            self.saved = {}

        def update_diagnostic(self, key, value):
            self.saved[key] = value

        def get_prediction(self, key):
            return {"run_id": "test"}

        def event(self, *args):
            pass

    class Worker:
        def __init__(self):
            self.entered = asyncio.Event()
            self.release = asyncio.Event()

        async def start(self):
            return {"test": True}

        async def analyze(self, r):
            self.entered.set()
            await self.release.wait()
            return {"status": "complete", "passed": True, "request_id": r["request_id"]}

        async def close(self):
            pass

    async def scenario():
        config = {
            "source": "live",
            "offline_ik": {"enabled": True},
            "max_age_s": 0.75,
            "max_skew_s": 0.2,
            "max_source_age_s": 1.0,
        }
        store = Store()
        manager = SimpleNamespace(
            config=config, store=store, generation=1, latest=None, storage_error=None
        )
        worker = Worker()
        checks = OfflineChecks(manager, worker)
        await checks.start()
        snapshot = DemoSource().read()
        snapshot.meta["joints"] = {
            "names": list(JOINT_NAMES),
            "positions": REFERENCE["joints"]["positions"],
            "source": snapshot.meta["sources"]["image"].copy(),
        }

        def prediction(key):
            return {"id": key, "actions": [[0] * 14] * 50, "offline_ik": {"status": "pending"}}

        first = prediction("first")
        manager.latest = first
        await checks.submit(first, snapshot, 1)
        await worker.entered.wait()
        await checks.submit(prediction("second"), snapshot, 1)
        third = prediction("third")
        await checks.submit(third, snapshot, 1)
        assert checks.queue.qsize() == 1 and store.saved["second"]["status"] == "superseded"
        manager.generation = 2
        manager.latest = prediction("new-run")
        worker.release.set()
        await asyncio.wait_for(checks.queue.join(), 2)
        assert (
            manager.latest["id"] == "new-run"
            and manager.latest["offline_ik"]["status"] == "pending"
        )
        assert store.saved["first"]["historical"] and store.saved["third"]["status"] == "superseded"
        await checks.close()

    asyncio.run(scenario())
