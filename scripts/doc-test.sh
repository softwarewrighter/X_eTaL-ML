#!/usr/bin/env bash
# Run every ## >> example in this repo's libraries (xetal doc --test):
# each prints what its doc comment says it prints. Run by the gate.
#   scripts/doc-test.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
xetal="$("$root/scripts/xetal.sh")"
path="$(ls -d libs/*/src work/libs/*/src | paste -sd: -)"
n=0
for f in libs/*/src/*.xtl libs/*/src/*.xtlm; do
  report="$(XETAL_PATH="$path" "$xetal" doc --test "$f" 2>&1)" || { echo "$report"; echo "doc-test: FAIL $f"; exit 1; }
  n=$((n + $(printf '%s\n' "$report" | grep -c ' \.\.\. ok$' || true)))
done
echo "doc-test: ok ($n examples)"
