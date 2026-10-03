#!/usr/bin/env bash
# Screenshot every demo with a web app from the built pages/ (run just
# pages first), with headless Chrome, into demos/<slug>/screenshot.png
# (900 px wide). The README of each demo shows it; the catalog uses it
# as the card's picture.
#   scripts/screenshots.sh [SLUG...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
chrome="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
[ -x "$chrome" ] || { echo "screenshots: no Chrome at $chrome (set CHROME)" >&2; exit 1; }
port=8099
"$root/scripts/serve-pages.sh" "$port" >/dev/null 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
sleep 1
if [ $# -gt 0 ]; then slugs=("$@"); else
  slugs=(); while IFS= read -r s; do [ -n "$s" ] && slugs+=("$s"); done < <("$root/scripts/demos.py" list)
fi
tmp="$(mktemp -d)"
for slug in "${slugs[@]}"; do
  [ -f "$root/demos/$slug/web/Cargo.toml" ] || continue
  # A watchdog: headless Chrome occasionally never returns on a page
  # that keeps animating; give each shot 60 seconds, then try once more.
  for try in 1 2; do
    "$chrome" --headless=new --disable-gpu --hide-scrollbars --window-size=1300,900 \
      --virtual-time-budget=12000 --screenshot="$tmp/$slug.png" \
      "http://127.0.0.1:$port/X_eTaL-ML/$slug/" >/dev/null 2>&1 &
    shot=$!
    ( sleep 60; kill "$shot" 2>/dev/null ) &
    dog=$!
    wait "$shot" 2>/dev/null || true
    kill "$dog" 2>/dev/null || true
    [ -s "$tmp/$slug.png" ] && break
    echo "screenshots: $slug timed out (try $try)" >&2
  done
  [ -s "$tmp/$slug.png" ] || { echo "screenshots: no picture for $slug" >&2; continue; }
  magick "$tmp/$slug.png" -resize 900x -colors 256 -strip -define png:compression-level=9 "$root/demos/$slug/screenshot.png"
  echo "screenshot: demos/$slug/screenshot.png"
done
rm -rf "$tmp"
