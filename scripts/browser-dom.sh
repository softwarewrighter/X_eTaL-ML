#!/usr/bin/env bash
# Load one demo's built page (pages/<slug>/) in headless Chrome, served
# as GitHub Pages serves it, let its WebAssembly run X_eTaL, and print
# the rendered DOM. A private profile per run; only this script's own
# server and Chrome are stopped.
#   scripts/browser-dom.sh SLUG [MS]   (MS: virtual time to run, default 6000)
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
slug="${1:?usage: browser-dom.sh SLUG [MS]}"
ms="${2:-6000}"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$chrome" ] || { echo "browser-dom: no Chrome at $chrome (set CHROME)" >&2; exit 1; }
[ -f "$root/pages/$slug/index.html" ] || { echo "browser-dom: no pages/$slug (just pages)" >&2; exit 1; }
port="$(python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')"
tmp="$(mktemp -d)"
site="$tmp/site"; mkdir -p "$site"; ln -s "$root/pages" "$site/X_eTaL-ML"
python3 -m http.server "$port" --bind 127.0.0.1 --directory "$site" >/dev/null 2>&1 &
server=$!
cleanup() { kill "$server" 2>/dev/null || true; wait "$server" 2>/dev/null || true; rm -rf "$tmp"; }
trap cleanup EXIT
for _ in $(seq 1 50); do curl -s -o /dev/null "http://127.0.0.1:$port/" && break; sleep 0.1; done
"$chrome" --headless=new --disable-gpu --no-first-run --no-default-browser-check \
  --user-data-dir="$tmp/profile" --virtual-time-budget="$ms" --dump-dom \
  "http://127.0.0.1:$port/X_eTaL-ML/$slug/" >"$tmp/dom" 2>/dev/null &
shot=$!
# An animated page keeps headless Chrome alive after the dump: stop it
# once the whole document has been written (or after 60 s).
for _ in $(seq 1 600); do
  grep -q '</html>' "$tmp/dom" 2>/dev/null && break
  kill -0 "$shot" 2>/dev/null || break
  sleep 0.1
done
kill "$shot" 2>/dev/null || true
wait "$shot" 2>/dev/null || true
cat "$tmp/dom"
