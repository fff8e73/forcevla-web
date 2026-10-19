#!/usr/bin/env bash
set -e
loop_build_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$loop_build_root"
sha256sum --check native_loop/vendor/SHA256SUMS
source /opt/kernelmind/apex/install/setup.bash
cmake -S native_loop -B build/loop -DCMAKE_BUILD_TYPE=Release
cmake --build build/loop -j2
