"""Isolated browser fixture: only synthetic inputs, model and plant."""

import os
from pathlib import Path

import uvicorn

from forcevla_web.app import create_app

root = Path(os.environ.get("LOOP_BROWSER_DATA", "artifacts/closed-loop-browser"))
root.mkdir(parents=True, exist_ok=True)
config = dict(
    source="demo",
    allowed_hosts=["127.0.0.1", "localhost"],
    max_age_s=0.75,
    max_skew_s=0.2,
    max_source_age_s=1,
    disconnect_grace_s=10,
    data_dir=str(root.resolve()),
    quota_mb=256,
    default_prompt="Synthetic closed-loop test",
    closed_loop=dict(
        enabled=True,
        integration_root="/home/user/forcevla-web/robot_control",
        simulation_rpc_s=0.25,
    ),
)
uvicorn.run(
    create_app(config),
    host="127.0.0.1",
    port=int(os.environ.get("LOOP_BROWSER_PORT", "8101")),
    access_log=False,
)
