#!/usr/bin/env bash
# Run a demo's program with the pinned xetal, in the demo's directory
# (this repo's libraries on XETAL_PATH),
# pictures ([]S_HOW) to work/draw/SLUG/. FILE defaults to SLUG.xtl.
#   scripts/run-demo.sh SLUG [FILE] [-- XETAL_RUN_FLAGS...]
#   scripts/run-demo.sh --echo SLUG [FILE]     # as a notebook
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
flags=()
if [ "${1:-}" = "--echo" ]; then flags+=(--echo); shift; fi
slug="${1:?usage: run-demo.sh [--echo] SLUG [FILE]}"
d="$root/demos/$slug"
[ -f "$d/demo.toml" ] || { echo "run-demo: no demo $slug" >&2; exit 1; }
file="${2:-$slug.xtl}"
[ -f "$d/$file" ] || { echo "run-demo: no $slug/$file" >&2; exit 1; }
if head -1 "$d/$file" | grep -q -- '--untyped'; then flags+=(--untyped); fi
xetal="$("$root/scripts/xetal.sh")"
mkdir -p "$root/work/draw/$slug"
# This repo's libraries on XETAL_PATH (relative to the demo's directory).
XETAL_PATH="$(cd "$root" && ls -d libs/*/src work/libs/*/src 2>/dev/null | sed 's#^#../../#' | paste -sd: -)"
export XETAL_PATH
cd "$d" && exec "$xetal" run ${flags[@]+"${flags[@]}"} --draw "$root/work/draw/$slug" "$file"
