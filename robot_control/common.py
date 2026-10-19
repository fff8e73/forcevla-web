"""Local configuration and durable reports for read-only integration tools."""

import hashlib
import json
import math
import shutil
from pathlib import Path
import time

CONTROL_ROOT = Path(__file__).resolve().parent
ROOT = CONTROL_ROOT.parent


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(path=None):
    config = json.loads(Path(path or ROOT / "config/robot.json").read_text())
    if not isinstance(config.get("prompt"), str) or not config["prompt"].strip():
        raise ValueError("prompt must be nonempty")
    for key, value in config["limits"].items():
        if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid observation budget: " + key)
    for key in [
        "camera_mapping_verified",
        "camera_transport_equivalence_verified",
        "physical_tcp_verified",
        "wrench_physical_semantics_verified",
    ]:
        if type(config.get(key)) is not bool:
            raise ValueError("Verification flags must be explicit booleans: " + key)
    shape = config["camera_decoded_shape"]
    if (
        len(shape) != 3
        or shape[2] != 3
        or any(type(x) is not int or x <= 0 for x in shape)
    ):
        raise ValueError("Invalid camera_decoded_shape")
    camera = config["expected_parameters"]["/quad_csi_quickview"]["webrtc"]
    if shape != [camera["output_height"], camera["output_width"], 3]:
        raise ValueError(
            "Decoded camera shape must agree with audited stream parameters"
        )
    return config


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    )
    tmp.replace(path)


def new_run(path, label):
    path = (
        Path(path)
        if path
        else ROOT / 'data/integration-runs' / (time.strftime("%Y%m%d-%H%M%S") + "-" + label)
    )
    path.mkdir(parents=True, exist_ok=False)
    source = path / "source"
    control = source / "robot_control"
    control.mkdir(parents=True)
    for item in [*CONTROL_ROOT.glob("*.py"), *CONTROL_ROOT.glob("*.sh")]:
        shutil.copy2(item, control / item.name)
    shutil.copy2(ROOT / "requirements-control.txt", source / "requirements-control.txt")
    shutil.copytree(ROOT / "tests/robot_control", source / "tests/robot_control", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(ROOT / "config", source / "config")
    shutil.copytree(CONTROL_ROOT / "diagnostics", control / "diagnostics", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("native", "native_loop"):
        if (CONTROL_ROOT / name).exists():
            shutil.copytree(CONTROL_ROOT / name, control / name, ignore=shutil.ignore_patterns("*.log"))
    if (CONTROL_ROOT / "vendor/daqp").exists():
        for name in ["include", "src"]:
            shutil.copytree(CONTROL_ROOT / "vendor/daqp" / name, control / "vendor/daqp" / name)
        shutil.copy2(CONTROL_ROOT / "vendor/daqp/LICENSE", control / "vendor/daqp/LICENSE")
    for name in ("offline_qp", "libcompatible_daqp.a", "loop/loop_motion"):
        if (CONTROL_ROOT / "build" / name).is_file():
            destination = control / "build" / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(CONTROL_ROOT / "build" / name, destination)
    return path


def source_manifest():
    return {str(p.relative_to(ROOT)): sha256(p) for p in sorted(CONTROL_ROOT.glob("*.py"))}
