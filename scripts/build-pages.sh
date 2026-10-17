#!/usr/bin/env bash
# Build the live site into pages/, which is not tracked: just publish
# (scripts/publish-pages.sh) makes it the gh-pages branch's only
# commit, which GitHub Pages serves (nothing is built on GitHub).
#   - every demo with a web app (demos/<slug>/web/) is built with trunk
#     into pages/<slug>/, served under /X_eTaL-ML/<slug>/;
#   - pages/libs/, the libraries' site (site/, trunk);
#   - pages/doc/, the cross-reference (scripts/doc-site.sh);
#   - pages/index.html, the catalog, from every demo.toml.
# A demo removed from demos/ loses its pages/<slug>/. With arguments
# (demo slugs, and libs for the libraries' site) only those are built
# and nothing is removed: the affected gate's build; the catalog and the
# cross-reference are rebuilt either way (seconds).
#   scripts/build-pages.sh [SLUG...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
base="/X_eTaL-ML"
"$root/scripts/py" "$root/scripts/demos.py" check
"$root/scripts/xetal.sh" >/dev/null   # the web apps build on work/xetal
mkdir -p "$root/pages"
touch "$root/pages/.nojekyll"
only=("$@")
# Whether SLUG is to be built: every one without arguments.
wanted() { [ ${#only[@]} -eq 0 ] || printf '%s\n' "${only[@]}" | grep -qx "$1"; }
keep=(libs)
while IFS= read -r slug; do
  [ -n "$slug" ] || continue
  web="$root/demos/$slug/web"
  wanted "$slug" || { [ -f "$web/Cargo.toml" ] && keep+=("$slug"); continue; }
  [ -f "$web/Cargo.toml" ] || continue
  keep+=("$slug")
  dist="$root/target/pages-dist/$slug"
  echo "==> trunk build $slug"
  # The app's own index.html is the trunk entry: demos/<slug>/web/index.html.
  (cd "$web" && trunk build --release --public-url "$base/$slug/" --dist "$dist")
  rsync -a --delete "$dist/" "$root/pages/$slug/"
  # The catalog card's picture, when the demo has one (just screenshots).
  if [ -f "$root/demos/$slug/screenshot.png" ]; then cp "$root/demos/$slug/screenshot.png" "$root/pages/$slug/"; fi
done < <("$root/scripts/py" "$root/scripts/demos.py" list)
# Drop pages/<slug>/ of demos that no longer have a web app (a full build).
[ ${#only[@]} -eq 0 ] && for d in "$root"/pages/*/; do
  [ -d "$d" ] || continue
  s="$(basename "$d")"
  printf '%s\n' ${keep[@]+"${keep[@]}"} | grep -qx "$s" || { echo "removing pages/$s/"; rm -rf "$d"; }
done
if wanted libs; then
  echo "==> trunk build site (the libraries)"
  (cd "$root/site" && trunk build --release --public-url "$base/libs/" --dist "$root/target/pages-dist/libs")
  rsync -a --delete "$root/target/pages-dist/libs/" "$root/pages/libs/"
fi
"$root/scripts/doc-site.sh"
cp "$root/images/xetal-logo-orange.png" "$root/images/favicon.ico" "$root/pages/"
"$root/scripts/py" "$root/scripts/build-catalog.py"
echo "pages/ built; just publish puts it live (after committing your changes)."
