#!/usr/bin/env bash
# The pre-commit gate, in two sizes.
#   scripts/gate.sh          affected: the cheap checks (the pinned X_eTaL,
#                            spelling, asks, rebind, literal c_at chains,
#                            the libraries' examples, markdown), then what
#                            the change touches (scripts/affected.py plans
#                            it from the files changed since HEAD): the
#                            demos and libraries changed or importing a
#                            changed library, their pages built and checked
#                            in the browser. Before every commit.
#   scripts/gate.sh --full   everything: the tooling self-tests, the shared
#                            page shell, every demo, every library, every
#                            page. After a pin, before publishing the whole
#                            site, and whenever the plan says all.
# Every step of 5 seconds or more prints its time; the end lists the
# slowest three. GATE_BASE names another base than HEAD for "changed";
# GATE_PLAN gives the plan itself (lines as affected.py prints them),
# to try the gate's paths.
#   scripts/gate.sh [--full]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
mode=affected
case "${1:-}" in
  --full) mode=full ;;
  "") ;;
  *) echo "usage: scripts/gate.sh [--full]" >&2; exit 2 ;;
esac
since=$SECONDS
times=""
last="start"
# A step's heading; the step before it is timed.
step() {
  if [ $((SECONDS - since)) -ge 5 ]; then printf '    (%ss)\n' $((SECONDS - since)); fi
  times="$times$((SECONDS - since)) $last\n"
  since=$SECONDS
  last="$*"
  printf '==> %s\n' "$*"
}

plan="all"
if [ "$mode" = affected ]; then
  "$root/scripts/py" "$root/scripts/affected.py" --self-test
  plan="${GATE_PLAN:-$("$root/scripts/py" "$root/scripts/affected.py")}"
  if grep -qx all <<< "$plan"; then mode=full; fi
fi
# The demos and libraries the plan names (every one in the full gate).
demos=(); libs=(); pages=0
if [ "$mode" = full ]; then
  pages=1
  while IFS= read -r s; do [ -n "$s" ] && demos+=("$s"); done < <("$root/scripts/py" "$root/scripts/demos.py" list)
  while IFS= read -r s; do [ -n "$s" ] && libs+=("$s"); done < <("$root/scripts/py" "$root/scripts/libs.py" list)
else
  while read -r kind name; do
    case "$kind" in demo) demos+=("$name") ;; lib) libs+=("$name") ;; pages) pages=1 ;; esac
  done <<< "$plan"
fi
if [ "$mode" = full ]; then echo "gate: full"; else
  printf 'gate: affected: %s demo(s), %s library(ies)%s\n' "${#demos[@]}" "${#libs[@]}" "$([ $pages = 1 ] && echo ', the pages' || true)"
fi

step "the pinned X_eTaL"
"$root/scripts/check-xetal.sh"
if [ "$mode" = full ]; then
  step "the demo tooling's self-test"
  "$root/scripts/selftest-demos.sh"
  step "the library tooling's self-test"
  "$root/scripts/selftest-libs.sh"
  # The shared shell of the demo pages (shared/microscope).
  step "shared/microscope"
  (cd "$root/shared/microscope" && cargo test -q >/dev/null 2>&1 && cargo check -q --target wasm32-unknown-unknown) \
    || { (cd "$root/shared/microscope" && cargo test -q); echo "FAIL: shared/microscope"; exit 1; }
  echo "ok: shared/microscope"
fi
if [ $pages = 1 ]; then
  # The libraries' site (site/): every library demo runs as its baseline recorded.
  step "the libraries' site"
  (cd "$root/site" && cargo test -q >/dev/null 2>&1 && cargo check -q --target wasm32-unknown-unknown) \
    || { (cd "$root/site" && cargo test -q); echo "FAIL: site"; exit 1; }
  echo "ok: site"
  # The site built fresh (not tracked; just publish publishes it), so the
  # browser checks below run against this checkout's pages: every page
  # in the full gate, the affected demos' and the libraries' otherwise.
  step "the pages built"
  if [ "$mode" = full ]; then "$root/scripts/build-pages.sh" >/dev/null
  else "$root/scripts/build-pages.sh" libs ${demos[@]+"${demos[@]}"} >/dev/null; fi
fi
if [ ${#demos[@]} -gt 0 ]; then
  step "the demos: ${demos[*]}"
  "$root/scripts/test-demos.sh" "${demos[@]}"
fi
if [ $pages = 1 ]; then
  # The libraries' site in a real browser (site/browser.txt).
  step "the libraries' site in the browser"
  "$root/scripts/browser-check.sh" libs >/dev/null || { "$root/scripts/browser-check.sh" libs; echo "FAIL: pages/libs"; exit 1; }
  echo "ok: pages/libs in the browser"
fi
if [ ${#libs[@]} -gt 0 ]; then
  step "the libraries: ${libs[*]}"
  "$root/scripts/test-libs.sh" "${libs[@]}"
  step "their doc examples"
  "$root/scripts/doc-test.sh" "${libs[@]}"
fi
step "the cheap checks"
"$root/scripts/py" "$root/scripts/check-examples.py"
# Our asks in step with X_eTaL's ledger at the pin.
"$root/scripts/py" "$root/scripts/check-asks.py" --self-test
"$root/scripts/py" "$root/scripts/check-asks.py"
# Each name bound once in its scope (X_eTaL lang-choices M1).
"$root/scripts/py" "$root/scripts/check-rebind.py" --self-test
"$root/scripts/py" "$root/scripts/check-rebind.py"
# Concise X_eTaL: number literals strand, so a shape is written
# 8 13 2 r_eshape x, never (8 c_at 13 c_at 2) r_eshape x (c_at is for names).
if git ls-files '*.xtl' '*.xtlm' '*.rs' '*.md' | grep -v -E '^(CHANGES.md|docs/plan.md|\.agentrail)' | xargs grep -n -E '\(([0-9]+ c_at )+[0-9]+\)' ; then
  echo "FAIL: literal c_at chains (write the numbers side by side)"; exit 1
fi
# American spellings only (the checker checks itself first).
"$root/scripts/py" "$root/scripts/check-spelling.py" --self-test
"$root/scripts/py" "$root/scripts/check-spelling.py"
md=(README.md CHANGES.md docs/plan.md docs/xetal-asks.md docs/gpu-asks.md docs/speed.md shared/microscope/README.md)
for f in demos/*/README.md libs/*/README.md libs/*/docs/README.md; do [ -e "$f" ] && md+=("$f"); done
for f in "${md[@]}"; do sw-markdown-checker -f "$f" >/dev/null || { sw-markdown-checker -f "$f"; exit 1; }; done
step "done"
printf 'gate: %s, %ss; the slowest:\n' "$mode" "$SECONDS"
printf "$times" | sort -rn | head -3 | sed 's/^/  /'
echo "gate: ok"
