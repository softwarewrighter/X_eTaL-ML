#!/usr/bin/env bash
# The base model and the tasks are reproducible: train/base.xtl, run
# again in a scratch copy under work/, writes the same tokens (the old
# task's and the shifted one's) and the same checkpoint, byte for byte,
# as data/ holds (fixed seeds; about half a minute). Run by
# scripts/test-demos.sh.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
xetal="$("$root/scripts/xetal.sh")"
scratch="$root/work/moe-base"
rm -rf "$scratch"; mkdir -p "$scratch/data"
cp "$here/train/base.xtl" "$scratch/base.xtl"
export XETAL_PATH="$(ls -d "$root"/libs/*/src "$root"/work/libs/*/src | paste -sd: -)"
(cd "$scratch" && "$xetal" run --seed 1 base.xtl >/dev/null)
for f in train.txt test.txt shift-train.txt shift-test.txt base.txt; do
  cmp -s "$here/data/$f" "$scratch/data/$f" || { echo "test.sh: data/$f differs from a fresh run of train/base.xtl" >&2; exit 1; }
done
echo "ok: the base model retrains to data/ exactly"
