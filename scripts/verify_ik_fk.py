"""Cross-check the isolated Pinocchio helper against the training MuJoCo model."""

import hashlib
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

root = Path(__file__).resolve().parents[1]
cfg = json.loads((root / "config/ik.json").read_text())
model = mujoco.MjModel.from_xml_path(cfg["model_path"])
data = mujoco.MjData(model)
proc = subprocess.Popen(
    [str(root / "native/build/offline_ik_worker"), str(root / "config/ik.json")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.DEVNULL,
    text=True,
)
hello = json.loads(proc.stdout.readline())
assert hello["ready"]
metadata = hello["metadata"]
names = metadata["joint_names"]
addresses = [model.joint(name).qposadr[0] for name in names]
reference = json.loads((root / "tests/fixtures/joint_fk_reference.json").read_text())["joints"][
    "positions"
]
rng = np.random.default_rng(14)
samples = [
    np.array(reference),
    *rng.uniform(metadata["joint_lower"], metadata["joint_upper"], size=(32, 14)),
]
position = []
orientation = []
try:
    for index, q in enumerate(samples):
        proc.stdin.write(
            json.dumps({"operation": "inspect", "seed": q.tolist(), "request_id": str(index)})
            + "\n"
        )
        proc.stdin.flush()
        result = json.loads(proc.stdout.readline())
        data.qpos[addresses] = q
        mujoco.mj_kinematics(model, data)
        for side in ["left", "right"]:
            body = model.body(side + "_tool").id
            position.append(
                float(np.linalg.norm(np.array(result["tcp"][side]["position"]) - data.xpos[body]))
            )
            a = Rotation.from_quat(result["tcp"][side]["quaternion_xyzw"])
            b = Rotation.from_matrix(data.xmat[body].reshape(3, 3))
            orientation.append(float((a.inv() * b).magnitude()))
finally:
    proc.stdin.close()
    proc.wait(timeout=3)
report = {
    "samples": len(samples),
    "tools": ["left_tool", "right_tool"],
    "model_sha256": hashlib.sha256(Path(cfg["model_path"]).read_bytes()).hexdigest(),
    "max_position_error_m": max(position),
    "max_orientation_error_rad": max(orientation),
    "position_gate_m": 1e-5,
    "orientation_gate_rad": 1e-5,
    "robot_motion_commands": False,
}
report["passed"] = max(position) < 1e-5 and max(orientation) < 1e-5
(root / "artifacts/offline-ik-20261014/fk-crosscheck.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
assert report["passed"]
