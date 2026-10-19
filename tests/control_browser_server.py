"""Local browser fixture: simulated feedback and fake Apex only, no hardware calls."""

import sys
from pathlib import Path

import uvicorn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_control_validation import ApexFake, Source

from forcevla_web.app import create_app
from forcevla_web.manager import Manager
from forcevla_web.policy import DemoPolicy
from forcevla_web.store import Store

root = Path("artifacts/control-browser-data")
root.mkdir(exist_ok=True)
config = {
    "source": "live",
    "allowed_hosts": ["127.0.0.1", "localhost"],
    "max_age_s": 0.75,
    "max_skew_s": 0.2,
    "max_source_age_s": 1,
    "disconnect_grace_s": 10,
    "data_dir": str(root),
    "quota_mb": 100,
    "default_prompt": "Synthetic browser test only",
    "control_validation": {"enabled": True, "poll_period_s": 0.02},
}
manager = Manager(config, Source(), DemoPolicy(), Store(root, 100))
app = create_app(config, manager, validation_backend=ApexFake())
uvicorn.run(app, host="127.0.0.1", port=8098, access_log=False)
