#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
# Source ROS under its usual non-nounset semantics.
set +u
source /opt/ros/jazzy/setup.bash
set -u
expected=653e22316c1d54ee7c71500f45f3ff7afaadad07
actual=$(git -C "$root/vendor/daqp" rev-parse HEAD)
if [[ "$actual" != "$expected" ]] || [[ -n "$(git -C "$root/vendor/daqp" status --porcelain)" ]]; then
  echo 'Pinned clean DAQP v0.7.2 source required; refusing an unreviewed source change.' >&2
  exit 1
fi
cmake -S "$root/native" -B "$root/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$root/build" -j2
