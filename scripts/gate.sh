#!/usr/bin/env bash
# The pre-commit gate: the vendored X_eTaL (scripts/check-vendor.sh),
# the demo and library tooling (self-tests), the shared page shell,
# every demo's tests, every library's tests and its page's examples,
# then ASCII-only markdown for the docs we own.
#   scripts/gate.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$root/scripts/check-vendor.sh"
"$root/scripts/selftest-demos.sh"
"$root/scripts/selftest-libs.sh"
# The shared shell of the demo pages (shared/microscope).
(cd "$root/shared/microscope" && cargo test -q >/dev/null 2>&1 && cargo check -q --target wasm32-unknown-unknown) \
  || { (cd "$root/shared/microscope" && cargo test -q); echo "FAIL: shared/microscope"; exit 1; }
echo "ok: shared/microscope"
"$root/scripts/test-demos.sh"
"$root/scripts/test-libs.sh"
"$root/scripts/check-examples.py"
md=(README.md CHANGES.md docs/plan.md docs/xetal-asks.md shared/microscope/README.md)
for f in demos/*/README.md libs/*/README.md libs/*/docs/README.md; do [ -e "$f" ] && md+=("$f"); done
for f in "${md[@]}"; do sw-markdown-checker -f "$f" >/dev/null || { sw-markdown-checker -f "$f"; exit 1; }; done
echo "gate: ok"
