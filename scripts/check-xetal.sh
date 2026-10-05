#!/usr/bin/env bash
# Check the pinned X_eTaL: the CLI builds, answers, reports the commit
# in XETAL_COMMIT, runs a demo and imports a standard library; the
# library used from X_eTaL-libraries (Check) loads; and a workspace
# outside work/xetal can use xetal-play natively and for wasm32 (as
# every demo's web app does).
#   scripts/check-xetal.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xetal="$("$root/scripts/xetal.sh")"
short="$(cut -c1-7 "$root/XETAL_COMMIT")"
got="$("$xetal" eval -e "'+ r_/_2 2 3 r_eshape r_ange 6")"
[ "$got" = "6 15" ] || { echo "check-xetal: eval gave '$got', expected '6 15'" >&2; exit 1; }
"$xetal" --version | grep -q "$short" || { echo "check-xetal: xetal --version does not report $short" >&2; "$xetal" --version >&2; exit 1; }
"$xetal" run "$root/work/xetal/demos/life.xtl" >/dev/null
got="$("$xetal" eval -e '"s:" u_se< "Stats"
s:m_ean 1 2 3 4')"
[ "$got" = "2.5" ] || { echo "check-xetal: Stats gave '$got', expected '2.5'" >&2; exit 1; }
got="$("$root/scripts/xt" eval -e '"k:" u_se< "Check"
6 k:i_s 2 * 3')"
[ "$got" = "ok" ] || { echo "check-xetal: Check (X_eTaL-libraries) gave '$got', expected 'ok'" >&2; exit 1; }
cd "$root/tools/xetal-probe"
cargo test -q >/dev/null 2>&1 || { cargo test; exit 1; }
cargo check -q --target wasm32-unknown-unknown
echo "check-xetal: ok ($short, libraries $(cut -c1-7 "$root/XETAL_LIBRARIES_COMMIT"))"
