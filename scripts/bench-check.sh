#!/usr/bin/env bash
# The ML speed check: every benchmark and demo program timed (best of
# RUNS, default 5) and compared with this machine's baseline,
# bench/baseline/HOST.tsv. A program more than BENCH_LIMIT percent
# (default 15) slower than its baseline, and slower by more than
# BENCH_FLOOR ms (default 15), fails; faster ones are listed too, so a
# fix upstream (ask M3, X_eTaL Saga 30) shows. --bless records the
# times now as the baseline (blessing a slowdown needs the user's
# approval; say so in the commit). As X_eTaL's own bench-check.
#   scripts/bench-check.sh [--bless] [RUNS]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
bless=false
if [ "${1:-}" = "--bless" ]; then bless=true; shift; fi
runs="${1:-5}"
limit="${BENCH_LIMIT:-15}"
floor="${BENCH_FLOOR:-15}"
baseline="$root/bench/baseline/$(hostname -s).tsv"
. "$root/scripts/bench-lib.sh"
was() { [ -f "$baseline" ] && awk -F '\t' -v p="$1" '$1 == p { print $2 }' "$baseline" || true; }
times="$(mktemp)"
failed=0
printf '%-36s %10s %9s %8s\n' "program" "baseline" "now" "change"
for p in "${programs[@]}"; do
  now="$(best "$p")"
  printf '%s\t%s\n' "$p" "$now" >> "$times"
  base="$(was "$p")"
  if [ -z "$base" ]; then printf '%-36s %10s %7sms %8s\n' "$p" "-" "$now" "new"; continue; fi
  change=$(( (now - base) * 100 / (base > 0 ? base : 1) ))
  mark=""
  if [ $(( now * 100 )) -gt $(( base * (100 + limit) )) ] && [ $(( now - base )) -gt "$floor" ]; then
    mark="  SLOWER"; failed=1
  elif [ $(( now * 100 )) -lt $(( base * (100 - limit) )) ] && [ $(( base - now )) -gt "$floor" ]; then
    mark="  faster"
  fi
  printf '%-36s %8sms %7sms %7s%%%s\n' "$p" "$base" "$now" "$change" "$mark"
done
if $bless; then
  mkdir -p "$(dirname "$baseline")"
  mv "$times" "$baseline"
  echo "bench-check: baseline recorded in bench/baseline/$(basename "$baseline")"
  exit 0
fi
rm -f "$times"
[ -f "$baseline" ] || { echo "bench-check: no baseline for $(hostname -s); record one with just bench-bless"; exit 2; }
if [ "$failed" -ne 0 ]; then
  echo "bench-check: slower than the baseline by more than $limit% (SLOWER above);"
  echo "file it in docs/xetal-asks.md, or with the user's approval record a new baseline (just bench-bless)"
  exit 1
fi
echo "bench-check: within $limit% of the baseline"
