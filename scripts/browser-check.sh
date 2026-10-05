#!/usr/bin/env bash
# Test one demo's built page in a real browser: headless Chrome loads
# pages/<slug>/ (scripts/browser-dom.sh), whose WebAssembly runs the
# demo's .xtl with the pinned X_eTaL; then the rendered page must
# show no X_eTaL error and every line of demos/<slug>/web/browser.txt
# (text the page only shows once X_eTaL has run and printed its
# arrays). Prints one line per check, the same each run, so reg-rs can
# keep it as a baseline (browser-<slug> in the demo's reg/).
#   scripts/browser-check.sh SLUG
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slug="${1:?usage: browser-check.sh SLUG}"
want="$root/demos/$slug/web/browser.txt"
[ -f "$want" ] || { echo "browser-check: no demos/$slug/web/browser.txt" >&2; exit 1; }
"$root/scripts/browser-dom.sh" "$slug" | python3 -c '
import html, re, sys
dom = sys.stdin.read()
want = [l.rstrip("\n") for l in open(sys.argv[1]) if l.strip() and not l.startswith("#")]
main = re.search(r"<main>(.*)</main>", dom, re.S)
text = html.unescape(re.sub(r"<[^>]+>", "", main.group(1))) if main else ""
ok = True
print("page: loaded" if "</html>" in dom else "page: NOT LOADED")
ok &= "</html>" in dom
errors = [m for m in ("X_eTaL stopped", "class=\"error\"", "panicked") if m in dom]
print("errors: " + (", ".join(errors) if errors else "none"))
ok &= not errors
for w in want:
    found = w in text
    print(("found: " if found else "MISSING: ") + w)
    ok &= found
sys.exit(0 if ok else 1)
' "$want"
