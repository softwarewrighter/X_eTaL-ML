#!/usr/bin/env bash
# Verify the deployed site after just publish: wait until the catalog
# at https://softwarewrighter.github.io/X_eTaL-ML/ reports this
# checkout's commit (GitHub Pages takes a minute), then load every
# demo's live page in headless Chrome and check, as the local browser
# test does, that X_eTaL ran there: no error, and every line of the
# demo's web/browser.txt shown.
#   scripts/check-live.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
base="https://softwarewrighter.github.io/X_eTaL-ML"
head="$(git rev-parse --short HEAD)"
live=""
for _ in $(seq 1 40); do
  live="$(curl -fsS "$base/?t=$(date +%s)" 2>/dev/null | grep -o 'sha [0-9a-f]\{7\}' | head -1 | cut -d' ' -f2 || true)"
  [ "$live" = "$head" ] && break
  sleep 6
done
[ "$live" = "$head" ] || { echo "check-live: the site reports commit '${live:-none}', not $head (just publish, or wait)" >&2; exit 1; }
echo "ok: catalog live at commit $head"
fail=0
while IFS= read -r slug; do
  [ -f "demos/$slug/web/browser.txt" ] || continue
  if out="$(XETAL_LIVE=1 scripts/browser-check.sh "$slug" 2>&1)"; then
    echo "ok: $base/$slug/ ($(printf '%s\n' "$out" | grep -c '^found:') checks)"
  else
    echo "FAIL: $base/$slug/"; printf '%s\n' "$out"; fail=1
  fi
done < <(scripts/demos.py list)
[ $fail = 0 ] && echo "check-live: ok" || { echo "check-live: FAILURES"; exit 1; }
