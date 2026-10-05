# Shared by scripts/bench.sh and bench-check.sh (sourced): the programs
# timed, and the best of $runs wall times of one, in milliseconds.
# The pinned xetal (release), every library on XETAL_PATH; a bench
# program runs from the repository root, a demo from its directory (so
# its data/ is found). Each prints little, so printing does not count.
xetal="$("$root/scripts/xetal.sh")"
export XETAL_DRAW="$root/work/bench-draw"
mkdir -p "$XETAL_DRAW"
programs=()
for p in "$root"/bench/*.xtl; do programs+=("bench/$(basename "$p")"); done
while IFS= read -r s; do
  [ -n "$s" ] && [ -f "$root/demos/$s/$s.xtl" ] && programs+=("demos/$s/$s.xtl")
done < <("$root/scripts/demos.py" list)

best() {
  local p="$1" dir file best="" t start end
  dir="$root/$(dirname "$p")"; file="$(basename "$p")"
  local path
  path="$(cd "$root" && ls -d libs/*/src work/libs/*/src 2>/dev/null | sed "s#^#$root/#" | paste -sd: -)"
  for _ in $(seq "$runs"); do
    start=$(python3 -c 'import time; print(time.monotonic_ns())')
    (cd "$dir" && XETAL_PATH="$path" "$xetal" run --seed 1 "$file" > /dev/null 2>&1)
    end=$(python3 -c 'import time; print(time.monotonic_ns())')
    t=$(( (end - start) / 1000000 ))
    if [ -z "$best" ] || [ "$t" -lt "$best" ]; then best=$t; fi
  done
  echo "$best"
}
