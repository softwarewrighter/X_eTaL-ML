#!/usr/bin/env bash
# Check the vendored X_eTaL: the CLI builds, answers, reports the
# vendored commit, runs a demo and imports a standard library; and a
# workspace outside vendor/xetal can use xetal-play natively and for
# wasm32 (as every demo's web app does).
#   scripts/check-vendor.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
xetal="$("$root/scripts/build-xetal.sh")"
sha="$(sed -n 's/^commit = "\(.......\).*/\1/p' "$root/vendor/xetal/VENDORED")"
got="$("$xetal" eval -e "'+ r_/_2 2 3 r_eshape r_ange 6")"
[ "$got" = "6 15" ] || { echo "check-vendor: eval gave '$got', expected '6 15'" >&2; exit 1; }
"$xetal" --version | grep -q "$sha" || { echo "check-vendor: xetal --version does not name $sha" >&2; "$xetal" --version >&2; exit 1; }
"$xetal" run "$root/vendor/xetal/demos/life.xtl" >/dev/null
got="$("$xetal" eval -e '"s:" u_se< "Stats"
s:m_ean 1 2 3 4')"
[ "$got" = "2.5" ] || { echo "check-vendor: Stats gave '$got', expected '2.5'" >&2; exit 1; }
# The libraries vendored from X_eTaL-libraries load and work here.
got="$("$root/scripts/xt" eval -e '"k:" u_se< "Check"
6 k:i_s 2 * 3')"
[ "$got" = "ok" ] || { echo "check-vendor: vendored Check gave '$got', expected 'ok'" >&2; exit 1; }
cd "$root/tools/vendor-probe"
cargo test -q >/dev/null 2>&1 || { cargo test; exit 1; }
cargo check -q --target wasm32-unknown-unknown
echo "check-vendor: ok ($sha)"
