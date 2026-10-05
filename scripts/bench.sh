#!/usr/bin/env bash
# Time the ML workloads (bench/*.xtl and every demo's program) with the
# vendored X_eTaL: each runs RUNS times (default 3) and the best wall
# time is printed as a markdown table, for docs/speed.md. Not part of
# the gate (timings depend on the machine).
#   scripts/bench.sh [RUNS]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs="${1:-3}"
. "$root/scripts/bench-lib.sh"
echo "| Program | Best of $runs (ms) |"
echo "| ------- | -------------- |"
for p in "${programs[@]}"; do echo "| \`$p\` | $(best "$p") |"; done
