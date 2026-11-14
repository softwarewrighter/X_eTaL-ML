#!/usr/bin/env bash
# Build the ML timeline (timeline/, forked from web-sw-cor24-macrolisp's
# lisp-timeline) into pages/ml-timeline/, served under
# /X_eTaL-ML/ml-timeline/: its native tests first (the events checked,
# the scale, the folds, the layout, the navigation and the hash), then
# trunk. It runs no X_eTaL, so it is not a demo; scripts/build-pages.sh
# calls this, and the gate checks the page in the browser
# (scripts/browser-check.sh ml-timeline, timeline/browser.txt).
#   scripts/build-timeline.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
src="$root/timeline"
dist="$root/target/pages-dist/ml-timeline"
echo "==> the ML timeline: tests"
(cd "$src" && cargo test -q -p ml-timeline >/dev/null 2>&1) || { (cd "$src" && cargo test -q -p ml-timeline); exit 1; }
echo "==> trunk build ml-timeline"
(cd "$src/crates/ml-timeline-web" && trunk build --release --public-url "/X_eTaL-ML/ml-timeline/" --dist "$dist")
mkdir -p "$root/pages/ml-timeline"
rsync -a --delete "$dist/" "$root/pages/ml-timeline/"
