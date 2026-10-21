#!/usr/bin/env bash
# The pipelines' expansions, pinned: what the Flow macro (and, in
# rosetta.xtl and diffusion.xtl, Equations) wrote, by the pinned xetal (reg-rs baselines
# expand-pipelines, expand-rosetta and expand-diffusion in reg/). Run by scripts/test-demos.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$here"
export XETAL="$("$root/scripts/xetal.sh")"
export XETAL_PATH="$(cd "$root" && ls -d libs/*/src work/libs/*/src | sed 's#^#../../#' | paste -sd: -)"
for name in expand-pipelines expand-rosetta expand-diffusion; do
  [ -f "$here/reg/$name.rgt" ] || { echo "test.sh: no reg/$name.rgt" >&2; exit 1; }
  REG_RS_DATA_DIR="$here/reg" reg-rs run -q -p "$name" || { REG_RS_DATA_DIR="$here/reg" reg-rs run -vv -p "$name" | tail -n +2; exit 1; }
done
