#!/usr/bin/env bash
# The pre-commit gate: the pinned X_eTaL (scripts/check-xetal.sh;
# it clones and builds on a fresh checkout),
# the demo and library tooling (self-tests), the shared page shell,
# every demo's tests, every library's tests and its page's examples,
# then ASCII-only markdown for the docs we own.
#   scripts/gate.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
"$root/scripts/check-xetal.sh"
"$root/scripts/selftest-demos.sh"
"$root/scripts/selftest-libs.sh"
# The shared shell of the demo pages (shared/microscope).
(cd "$root/shared/microscope" && cargo test -q >/dev/null 2>&1 && cargo check -q --target wasm32-unknown-unknown) \
  || { (cd "$root/shared/microscope" && cargo test -q); echo "FAIL: shared/microscope"; exit 1; }
echo "ok: shared/microscope"
# The site built fresh (not tracked; just publish publishes it), so the
# browser checks below run against this checkout's pages.
"$root/scripts/build-pages.sh" >/dev/null
"$root/scripts/test-demos.sh"
"$root/scripts/test-libs.sh"
"$root/scripts/doc-test.sh"
"$root/scripts/check-examples.py"
# Concise X_eTaL: number literals strand, so a shape is written
# 8 13 2 r_eshape x, never (8 c_at 13 c_at 2) r_eshape x (c_at is for names).
if git ls-files '*.xtl' '*.xtlm' '*.rs' '*.md' | grep -v -E '^(CHANGES.md|docs/plan.md|\.agentrail)' | xargs grep -n -E '\(([0-9]+ c_at )+[0-9]+\)' ; then
  echo "FAIL: literal c_at chains (write the numbers side by side)"; exit 1
fi
# American spellings only (the checker checks itself first).
"$root/scripts/check-spelling.py" --self-test
"$root/scripts/check-spelling.py"
md=(README.md CHANGES.md docs/plan.md docs/xetal-asks.md docs/speed.md shared/microscope/README.md)
for f in demos/*/README.md libs/*/README.md libs/*/docs/README.md; do [ -e "$f" ] && md+=("$f"); done
for f in "${md[@]}"; do sw-markdown-checker -f "$f" >/dev/null || { sw-markdown-checker -f "$f"; exit 1; }; done
echo "gate: ok"
