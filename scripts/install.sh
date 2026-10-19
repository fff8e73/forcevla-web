#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv sync --frozen
uv venv --python /usr/bin/python3 --system-site-packages .ros-venv
uv pip install --python .ros-venv/bin/python -r requirements-collector.txt
uv venv --python /usr/bin/python3 --system-site-packages .control-venv
uv pip install --python .control-venv/bin/python -r requirements-control.txt
npm ci --ignore-scripts

bash scripts/build_ik.sh

bash robot_control/build_native.sh
bash robot_control/build_loop.sh
