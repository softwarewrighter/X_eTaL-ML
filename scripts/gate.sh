#!/usr/bin/env bash
# The pre-commit gate. For now: ASCII-only markdown for the docs we
# own. Later steps add the vendored X_eTaL, the tooling self-tests,
# every demo's and library's tests, and the pages.
#   scripts/gate.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
md=(README.md CHANGES.md docs/plan.md docs/xetal-asks.md)
for f in demos/*/README.md libs/*/README.md libs/*/docs/README.md; do [ -e "$f" ] && md+=("$f"); done
for f in "${md[@]}"; do sw-markdown-checker -f "$f" >/dev/null || { sw-markdown-checker -f "$f"; exit 1; }; done
echo "gate: ok"
