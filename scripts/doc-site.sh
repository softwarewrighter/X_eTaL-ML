#!/usr/bin/env bash
# Build the cross-reference site (xetal doc --out, as X_eTaL's own
# /doc) into pages/doc: this repo's libraries (libs/*/src), their demo
# programs and every demo's programs, each file a page with its ##
# doc comments, ### sections, ## >> examples, its source drawn
# decorated and every name linked (into NN, Net and the built-ins); an
# index and a search. Run by scripts/build-pages.sh (just pages) and by
# `just doc`. Run from the repository root with relative library
# paths, so page names carry no local path.
#   scripts/doc-site.sh             # into pages/doc
#   XETAL_DOC_OUT=DIR scripts/doc-site.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
xetal="$("$root/scripts/xetal.sh")"
out="${XETAL_DOC_OUT:-pages/doc}"
path="$(ls -d libs/*/src work/libs/*/src | paste -sd: -)"
files=(libs/*/src/*.xtl libs/*/src/*.xtlm libs/*/demos/*.xtl)
for f in demos/*/*.xtl; do
  case "$f" in demos/_template/*) ;; *) files+=("$f") ;; esac
done
rm -rf "$out"
XETAL_PATH="$path" "$xetal" doc --out "$out" "${files[@]}" > /dev/null
echo "doc: $(find "$out" -name '*.html' | wc -l | tr -d ' ') pages in $out"
