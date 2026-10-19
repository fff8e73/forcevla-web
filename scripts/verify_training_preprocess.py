"""Compare the V1 transform to the independently maintained training converter."""

import ast
import hashlib
import io
import json
from pathlib import Path

import av
import cv2
import numpy as np
from PIL import Image

from forcevla_web.collector import split_tile
from forcevla_web.observation import CAMERAS

path = Path("/home/user/forcevla-conversion-20260920/export/convert_bimanual.py")
source = path.read_text()
function = next(
    n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "encode_tiles"
)
namespace = {"cv2": cv2, "np": np, "Image": Image, "io": io}
exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
with np.load(
    "artifacts/stability-20261014/image-contract/paired-rgb.npz", allow_pickle=False
) as arrays:
    rgb = arrays["record"].copy()
actual = split_tile(rgb, (1080, 1920, 3))
encoded = namespace["encode_tiles"](av.VideoFrame.from_ndarray(rgb, format="rgb24"))
results = {}
for key, data in zip(CAMERAS, encoded):
    expected = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    results[key] = {
        "equal": bool(np.array_equal(actual[key], expected)),
        "max_abs": int(np.abs(actual[key].astype(int) - expected.astype(int)).max()),
        "shape": list(actual[key].shape),
    }
report = {
    "training_converter": str(path),
    "training_converter_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "scope": "same canonical 1920x1080 RGB input under current libraries; does not assert identical H264 encodings",
    "views": results,
    "passed": all(v["equal"] for v in results.values()),
}
Path("artifacts/stability-20261014/training-preprocess.json").write_text(
    json.dumps(report, indent=2) + "\n"
)
print(json.dumps(report, indent=2))
assert report["passed"]
