#!/usr/bin/env bash
# Test the demos (every demos/<slug>/ but the template), or the named ones:
#   - each top-level <slug>/*.xtl is a reg-rs baseline in <slug>/reg/
#     (this repo's libraries on XETAL_PATH, so it can import them)
#     (cli-<name>.rgt, .out, .err): run with the vendored xetal (--seed
#     1, pictures to work/draw/<slug>/), its stdout, stderr and exit
#     code must match;
#   - web/ (a Cargo workspace), when present: cargo test, and cargo
#     check for wasm32 (the browser build);
#   - web/browser.txt, when present: the built page in headless Chrome
#     (scripts/browser-check.sh), a reg-rs baseline browser-<slug>;
#   - test.sh, when present and executable: run it.
# XETAL_BLESS=1 accepts the current output as the baselines instead
# (creating missing ones; review the diff!).
# XETAL_DEMOS_DIR overrides demos/ (scripts/selftest-demos.sh uses it).
#   scripts/test-demos.sh [SLUG...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$root/scripts/demos.py" check
xetal="$("$root/scripts/build-xetal.sh")"
if [ $# -gt 0 ]; then slugs=("$@"); else
  slugs=(); while IFS= read -r s; do [ -n "$s" ] && slugs+=("$s"); done < <("$root/scripts/demos.py" list)
fi
command -v reg-rs >/dev/null || { echo "test: reg-rs not found on PATH" >&2; exit 127; }
# Each demo's reg-rs baselines live in its own reg/: cli-NAME runs
# NAME.xtl with the vendored xetal (from the demo's directory, so the
# command reads the same anywhere); the .rgt (command, exit code) and
# the .out / .err are committed, the .tdb cache is not.
export XETAL="$xetal"
# This repo's libraries on XETAL_PATH, as paths relative to a demo's
# directory (every demo is demos/<slug>/), so `u_se<` finds them and
# baselines do not depend on where the repository is checked out.
XETAL_PATH="$(cd "$root" && ls -d libs/*/src vendor/xetal-libraries/libs/*/src 2>/dev/null | sed 's#^#../../#' | paste -sd: -)"
export XETAL_PATH
# bless DIR DATA NAME COMMAND DESC: make (or remake) baseline NAME in
# DATA from the current output, running COMMAND from DIR. An existing
# baseline keeps its command and description; remaking (rather than
# reg-rs rebase) also records a changed exit code.
bless() {
  local dir="$1" data="$2" name="$3" cmd="$4" desc="$5"
  mkdir -p "$data"
  if [ -f "$data/$name.rgt" ]; then
    cmd="$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb"))["command"])' "$data/$name.rgt")"
    desc="$(python3 -c 'import sys, tomllib; print(tomllib.load(open(sys.argv[1], "rb")).get("desc", ""))' "$data/$name.rgt")"
    rm -f "$data/$name".*
  fi
  (cd "$dir" && REG_RS_DATA_DIR="$data" reg-rs create -t "$name" -c "$cmd" --desc "$desc" >/dev/null)
}
fail=0; n=0
for slug in ${slugs[@]+"${slugs[@]}"}; do
  d="${XETAL_DEMOS_DIR:-$root/demos}/$slug"
  [ -f "$d/demo.toml" ] || { echo "test: no demo $slug" >&2; exit 1; }
  n=$((n + 1))
  export XETAL_DRAW="$root/work/draw/$slug"; mkdir -p "$XETAL_DRAW"
  for prog in "$d"/*.xtl; do
    [ -e "$prog" ] || continue
    name="cli-$(basename "$prog" .xtl)"
    reg() { (cd "$d" && REG_RS_DATA_DIR="$d/reg" reg-rs "$@"); }
    if [ "${XETAL_BLESS:-}" = 1 ]; then
      bless "$d" "$d/reg" "$name" "\"\$XETAL\" run --seed 1 --draw \"\$XETAL_DRAW\" $(basename "$prog")" \
        "$slug: $(basename "$prog") run by the vendored xetal CLI"
      echo "blessed: $slug/$name"; continue
    fi
    if [ ! -f "$d/reg/$name.rgt" ]; then
      echo "FAIL: $slug/$name: no reg/$name.rgt (XETAL_BLESS=1 to create)"; fail=1
    elif reg run -q -p "$name"; then
      echo "ok: $slug/$name"
    else
      echo "FAIL: $slug/$name"; reg run -vv -p "$name" | tail -n +2; fail=1
    fi
  done
  if [ -f "$d/web/Cargo.toml" ]; then
    (cd "$d/web" && cargo test -q --workspace >/dev/null 2>&1) \
      && echo "ok: $slug/web" || { echo "FAIL: $slug/web"; (cd "$d/web" && cargo test -q --workspace) || true; fail=1; }
    (cd "$d/web" && cargo check -q --target wasm32-unknown-unknown >/dev/null 2>&1) \
      && echo "ok: $slug/web (wasm32)" || { echo "FAIL: $slug/web (wasm32)"; (cd "$d/web" && cargo check -q --target wasm32-unknown-unknown) || true; fail=1; }
  fi
  # The page in a real browser (scripts/browser-check.sh), against the
  # built pages/<slug>/: a reg-rs baseline browser-<slug> in reg/.
  # XETAL_BROWSER=0 skips it (no Chrome, or pages/ not built).
  if [ -f "$d/web/browser.txt" ] && [ "${XETAL_BROWSER:-1}" != 0 ]; then
    name="browser-$slug"
    breg() { (cd "$root" && REG_RS_DATA_DIR="$d/reg" reg-rs "$@"); }
    if [ "${XETAL_BLESS:-}" = 1 ]; then
      # A browser baseline records a passing check only.
      if ! (cd "$root" && scripts/browser-check.sh "$slug" >/dev/null); then
        echo "FAIL: $slug/$name: the browser check fails, not blessed:"; (cd "$root" && scripts/browser-check.sh "$slug") || true; fail=1
      else
        bless "$root" "$d/reg" "$name" "scripts/browser-check.sh $slug" \
          "$slug: the built page run in headless Chrome shows X_eTaL's results"
        echo "blessed: $slug/$name"
      fi
    elif [ ! -f "$d/reg/$name.rgt" ]; then
      echo "FAIL: $slug/$name: no reg/$name.rgt (XETAL_BLESS=1 to create)"; fail=1
    elif breg run -q -p "$name"; then
      echo "ok: $slug/$name"
    else
      echo "FAIL: $slug/$name"; breg run -vv -p "$name" | tail -n +2; fail=1
    fi
  fi
  if [ -x "$d/test.sh" ]; then
    "$d/test.sh" && echo "ok: $slug/test.sh" || { echo "FAIL: $slug/test.sh"; fail=1; }
  fi
done
echo "test-demos: $n demo(s)$([ $fail = 0 ] && echo ', all passed' || echo ', FAILURES')"
exit $fail
