#!/usr/bin/env bash
# Start a demo sub-project from demos/_template: demos/SLUG/ with its
# demo.toml, README.md, SLUG.xtl and its reg-rs baseline reg/cli-SLUG.
#   scripts/new-demo.sh SLUG "Title"
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="${XETAL_DEMOS_DIR:-$root/demos}"
slug="${1:?usage: new-demo.sh SLUG \"Title\"}"
title="${2:?usage: new-demo.sh SLUG \"Title\"}"
[[ "$slug" =~ ^[a-z][a-z0-9-]*$ ]] || { echo "new-demo: slug must be lowercase letters, digits and -" >&2; exit 1; }
dest="$base/$slug"
[ ! -e "$dest" ] || { echo "new-demo: $dest exists" >&2; exit 1; }
cp -R "$root/demos/_template" "$dest"
mv "$dest/__SLUG__.xtl" "$dest/$slug.xtl"
mv "$dest/reg/cli-__SLUG__.out" "$dest/reg/cli-$slug.out"
mv "$dest/reg/cli-__SLUG__.rgt" "$dest/reg/cli-$slug.rgt"
chmod +x "$dest/$slug.xtl"
esc="$(printf '%s' "$title" | sed 's/[&|\\]/\\&/g')"
for f in "$dest/demo.toml" "$dest/README.md" "$dest/$slug.xtl" "$dest/reg/cli-$slug.rgt"; do
  sed -i '' -e "s|__SLUG__|$slug|g" -e "s|__TITLE__|$esc|g" "$f" 2>/dev/null \
    || sed -i -e "s|__SLUG__|$slug|g" -e "s|__TITLE__|$esc|g" "$f"
done
echo "new demo: $dest (edit demo.toml, README.md and $slug.xtl; just test-demo $slug)"
