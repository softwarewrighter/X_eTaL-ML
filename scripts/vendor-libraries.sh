#!/usr/bin/env bash
# Vendor libraries from X_eTaL-libraries: replace vendor/xetal-libraries/
# with the src/ of the named libraries (default: Check) at a COMMITTED
# ref of that repository (never its working tree), recorded in
# vendor/xetal-libraries/VENDORED. They go on XETAL_PATH beside this
# repo's own libs/ (scripts/xt, test-demos.sh, run-demo.sh) and into
# the pages' library store (shared/microscope).
#   scripts/vendor-libraries.sh [REF] [Name...]
#   XETAL_LIBRARIES_REPO=/path scripts/vendor-libraries.sh
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
repo="${XETAL_LIBRARIES_REPO:-$root/../X_eTaL-libraries}"
ref="${1:-HEAD}"; shift || true
names=("$@"); [ ${#names[@]} -gt 0 ] || names=(Check)
sha="$(git -C "$repo" rev-parse --verify "$ref^{commit}")"
paths=(); for n in "${names[@]}"; do paths+=("libs/$n/src"); done
dest="$root/vendor/xetal-libraries"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
git -C "$repo" archive --format=tar "$sha" -- "${paths[@]}" LICENSE COPYRIGHT | tar -x -C "$tmp"
mkdir -p "$dest"
rsync -a --delete --exclude=VENDORED "$tmp/" "$dest/"
{
  echo "repository = \"https://github.com/softwarewrighter/X_eTaL-libraries\""
  echo "commit = \"$sha\""
  echo "subject = \"$(git -C "$repo" log -1 --format=%s "$sha" | sed 's/"/\\"/g')\""
  echo "committed = \"$(git -C "$repo" log -1 --format=%cI "$sha")\""
  echo "vendored = \"$(date -u +%Y-%m-%dT%H:%M:%SZ)\""
  echo "libraries = [$(printf '"%s", ' "${names[@]}" | sed 's/, $//')]"
} > "$dest/VENDORED"
echo "vendored X_eTaL-libraries ${sha:0:7} (${names[*]}) into vendor/xetal-libraries/"
