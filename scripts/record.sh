#!/usr/bin/env bash
# Record the demos that run only at the command line (a demo with an
# interactive page is shown live instead, and has no tape): each demos/<slug>/<slug>.tape
# is played by VHS (charmbracelet/vhs; it types `just tour SLUG` in a
# fresh shell at the repository root), and the GIF it writes to
# work/record/ becomes demos/<slug>/recording.webp, the animated
# picture the demo's README and the recorded-demos page show: ffmpeg
# takes it to 8 frames a second and 960 pixels wide (VHS writes 25 a
# second whatever the tape says), gif2webp (libwebp) encodes it.
# All demos with a tape, or the named ones; --encode re-encodes the
# GIFs already in work/record/ without recording again.
#   scripts/record.sh [--encode] [SLUG...]
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$root"
encode_only=0
if [ "${1:-}" = --encode ]; then encode_only=1; shift; fi
for tool in vhs ffmpeg gif2webp; do
  command -v "$tool" >/dev/null || { echo "record: $tool not found (brew install vhs ffmpeg webp)" >&2; exit 127; }
done
if [ $# -gt 0 ]; then slugs=("$@"); else
  slugs=(); for t in demos/*/*.tape; do [ -e "$t" ] && slugs+=("$(basename "$(dirname "$t")")"); done
fi
"$root/scripts/build-xetal.sh" >/dev/null   # built before the tape starts typing
mkdir -p work/record
for slug in ${slugs[@]+"${slugs[@]}"}; do
  tape="demos/$slug/$slug.tape"
  [ -f "$tape" ] || { echo "record: no $tape" >&2; exit 1; }
  if [ $encode_only = 0 ]; then
    rm -f "work/record/$slug.gif"
    vhs "$tape" >/dev/null 2>&1
  fi
  [ -s "work/record/$slug.gif" ] || { echo "record: no work/record/$slug.gif (vhs failed?)" >&2; exit 1; }
  ffmpeg -loglevel error -y -i "work/record/$slug.gif" \
    -vf "fps=8,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse=dither=none" \
    -loop 0 "work/record/$slug-8fps.gif"
  gif2webp -quiet -lossy -q 50 -m 6 "work/record/$slug-8fps.gif" -o "demos/$slug/recording.webp"
  echo "recorded: demos/$slug/recording.webp ($(($(wc -c < "demos/$slug/recording.webp") / 1024)) KB)"
done
