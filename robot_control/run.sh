#!/usr/bin/env bash
set -e
control_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
project_root=$(cd "$control_root/.." && pwd -P)
source /opt/ros/jazzy/setup.bash
source /opt/kernelmind/apex/install/setup.bash
export ROS_DOMAIN_ID=24
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}:/opt/kernelmind/venv/lib/python3.12/site-packages"
cd "$project_root"
case "${1:-}" in
  preflight) entry=run_preflight ;;
  observe) entry=run_observe ;;
  predict) entry=run_predict ;;
  prepare) entry=run_prepare ;;
  first10) entry=run_first10 ;;
  control) entry=run_control ;;
  closed-loop) entry=run_closed_loop ;;
  planner-loop) entry=run_planner_loop ;;
  stage-b-sandbox) entry=diagnostics.stage_b_sandbox ;;
  stage-d-sandbox) entry=diagnostics.stage_d_sandbox ;;
  test) shift; exec "$project_root/.control-venv/bin/python" -m unittest discover -s "$project_root/tests/robot_control" "$@" ;;
  *) echo 'Usage: robot_control/run.sh {preflight|observe|predict|prepare|first10|control|closed-loop|planner-loop|stage-b-sandbox|stage-d-sandbox|test} [arguments]' >&2; exit 2 ;;
esac
shift
exec "$project_root/.control-venv/bin/python" -m "robot_control.$entry" "$@"
