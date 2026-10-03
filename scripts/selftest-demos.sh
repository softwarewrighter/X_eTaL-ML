#!/usr/bin/env bash
# Test the demo tooling itself in a scratch demos directory: new-demo
# makes a demo that passes; a wrong reg-rs baseline fails; an unexpected
# error fails; XETAL_BLESS=1 repairs it; a program without a baseline
# fails until blessed; the catalog shows it (escaped,
# no live link without a web app); a bad demo.toml is rejected.
#   scripts/selftest-demos.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XETAL_DEMOS_DIR="$(mktemp -d)"; export XETAL_DEMOS_DIR
trap 'rm -rf "$XETAL_DEMOS_DIR"' EXIT
t="$root/scripts/test-demos.sh"
expect() { # expect pass|fail DESCRIPTION
  if "$t" probe >/dev/null 2>&1; then got=pass; else got=fail; fi
  [ "$got" = "$1" ] || { echo "selftest: $2: expected $1, got $got" >&2; "$t" probe || true; exit 1; }
}
"$root/scripts/new-demo.sh" probe "Probe & Co" >/dev/null
grep -q 'title = "Probe & Co"' "$XETAL_DEMOS_DIR/probe/demo.toml"
expect pass "a fresh demo"
echo 56 > "$XETAL_DEMOS_DIR/probe/reg/cli-probe.out"
expect fail "a wrong baseline"
XETAL_BLESS=1 "$t" probe >/dev/null
expect pass "after blessing"
printf '#!/usr/bin/env xetal\n1 +\n' > "$XETAL_DEMOS_DIR/probe/probe.xtl"
expect fail "an unexpected error"
XETAL_BLESS=1 "$t" probe >/dev/null
grep -q 'exit_code = 1' "$XETAL_DEMOS_DIR/probe/reg/cli-probe.rgt" || [ -s "$XETAL_DEMOS_DIR/probe/reg/cli-probe.err" ]
expect pass "an expected error"
printf '#!/usr/bin/env xetal\n2 * 3\n' > "$XETAL_DEMOS_DIR/probe/second.xtl"
expect fail "a program without a baseline"
XETAL_BLESS=1 "$t" probe >/dev/null
[ -f "$XETAL_DEMOS_DIR/probe/reg/cli-second.rgt" ]
expect pass "after blessing the new program"
"$root/scripts/build-catalog.py" "$XETAL_DEMOS_DIR/index.html" >/dev/null
grep -q '<h2>Probe &amp; Co</h2>' "$XETAL_DEMOS_DIR/index.html" \
  || { echo "selftest: the catalog has no card for the demo" >&2; exit 1; }
! grep -q 'href="probe/"' "$XETAL_DEMOS_DIR/index.html" \
  || { echo "selftest: the catalog links a demo with no web app" >&2; exit 1; }
sed -i '' 's/^status = .*/status = "bogus"/' "$XETAL_DEMOS_DIR/probe/demo.toml"
expect fail "a bad status"
echo "selftest-demos: ok"
