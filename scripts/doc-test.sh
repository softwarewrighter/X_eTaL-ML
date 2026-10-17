#!/usr/bin/env bash
# Run every ## >> example in this repo's libraries (xetal doc --test),
# or in the named ones: each prints what its doc comment says it
# prints. Run by the gate.
#   scripts/doc-test.sh [Name...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
xetal="$("$root/scripts/xetal.sh")"
path="$(ls -d libs/*/src work/libs/*/src | paste -sd: -)"
n=0
files=()
if [ $# -gt 0 ]; then for name in "$@"; do files+=(libs/"$name"/src/*.xtl libs/"$name"/src/*.xtlm); done
else files=(libs/*/src/*.xtl libs/*/src/*.xtlm); fi
for f in "${files[@]}"; do
  [ -e "$f" ] || continue
  report="$(XETAL_PATH="$path" "$xetal" doc --test "$f" 2>&1)" || { echo "$report"; echo "doc-test: FAIL $f"; exit 1; }
  n=$((n + $(printf '%s\n' "$report" | grep -c ' \.\.\. ok$' || true)))
done
echo "doc-test: ok ($n examples)"
