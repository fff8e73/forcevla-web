#!/usr/bin/env bash
set -e
project_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)
exec "$project_root/robot_control/run.sh" "$@"
