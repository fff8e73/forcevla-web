#!/usr/bin/env bash
set -e
native_loop_root=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source /opt/kernelmind/apex/install/setup.bash
exec "$native_loop_root/build/loop/loop_motion" "$@"
