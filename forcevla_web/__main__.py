import argparse
import fcntl
import json
import os
import signal
import subprocess
from pathlib import Path

import uvicorn

from forcevla_web.app import create_app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/local.json")
    parser.add_argument(
        "--demo", action="store_true", help="Synthetic input AND policy, never real RPC"
    )
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if args.demo:
        config["source"] = "demo"
        config["data_dir"] += "-demo"
    if args.port:
        config["port"] = args.port
    root = Path(config["data_dir"])
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "instance.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("A ForceVLA web server already owns this data directory")
    collector = None
    if config["source"] == "live":
        env = os.environ.copy()
        env["ROS_DOMAIN_ID"] = str(config["ros_domain_id"])
        env["PYTHONPATH"] = str(Path(__file__).parent.parent)
        env.pop("VIRTUAL_ENV", None)
        collector = subprocess.Popen(
            [
                "/bin/bash",
                "-c",
                'source "$1" && exec "$3" -m forcevla_web.collector --config "$2"',
                "forcevla-collector",
                config["ros_setup"],
                str(config_path),
                str(Path(__file__).parent.parent / ".ros-venv/bin/python"),
            ],
            env=env,
            start_new_session=True,
        )
    try:
        uvicorn.run(
            create_app(config),
            host=config["host"],
            port=config["port"],
            log_level="info",
            access_log=False,
        )
    finally:
        if collector and collector.poll() is None:
            os.killpg(collector.pid, signal.SIGINT)
            try:
                collector.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(collector.pid, signal.SIGKILL)
                collector.wait()
        lock.close()


if __name__ == "__main__":
    main()
