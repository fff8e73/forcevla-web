#!/usr/bin/env bash
set -e
web_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
source /opt/ros/jazzy/setup.bash
source /opt/kernelmind/apex/install/setup.bash
export ROS_DOMAIN_ID=24
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$web_root${PYTHONPATH:+:$PYTHONPATH}"
exec "$web_root/.ros-venv/bin/python" -m forcevla_web.gripper_initial "$@"
