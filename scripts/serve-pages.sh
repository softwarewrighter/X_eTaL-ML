#!/usr/bin/env bash
# Serve the built pages/ locally as GitHub Pages will, under
# /X_eTaL-ML/: http://127.0.0.1:PORT/X_eTaL-ML/ (default 8098).
#   scripts/serve-pages.sh [PORT]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
port="${1:-8098}"
site="$root/target/serve-pages"
mkdir -p "$site"
ln -sfn "$root/pages" "$site/X_eTaL-ML"
echo "serving http://127.0.0.1:$port/X_eTaL-ML/ (Ctrl-C stops)"
exec python3 -m http.server "$port" --bind 127.0.0.1 --directory "$site"
