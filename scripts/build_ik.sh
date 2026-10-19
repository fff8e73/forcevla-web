#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
cmake -S native -B native/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/ros/jazzy
cmake --build native/build -j2
