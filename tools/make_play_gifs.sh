#!/usr/bin/env bash
# Turn the Play In Editor screenshots into the README GIFs in docs/media/.
# Run after unreal/playtest.py (and the aerial shots, if any) in the project.
#   tools/make_play_gifs.sh [project dir]   (default: ~/Documents/Unreal Projects/RedRocks)
set -euo pipefail
PROJ="${1:-$HOME/Documents/Unreal Projects/RedRocks}"
SHOTS="$PROJ/Saved/Screenshots/MacEditor"
OUT="$(cd "$(dirname "$0")/.." && pwd)/docs/media"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

gif() {  # gif <output name> <frame>...  (one frame every ~1.7 s, 960 px wide)
  local name="$1"; shift
  local i=0
  for f in "$@"; do
    [ -f "$SHOTS/$f.png" ] || { echo "missing $SHOTS/$f.png" >&2; return 1; }
    cp "$SHOTS/$f.png" "$TMP/$name-$(printf %02d $i).png"; i=$((i + 1))
  done
  ffmpeg -loglevel error -y -framerate 0.6 -i "$TMP/$name-%02d.png" \
    -vf "scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=192[p];[b][p]paletteuse=dither=sierra2_4a" \
    -loop 0 "$OUT/$name.gif"
  echo "wrote $OUT/$name.gif"
}

gif redrocks_play_tour redrocks_play_spawn redrocks_play_walked \
  redrocks_play_look_000 redrocks_play_look_090 redrocks_play_look_180 redrocks_play_look_270
aerial=()
for f in dbg_with_terrain dbg_scan_only; do [ -f "$SHOTS/$f.png" ] && aerial+=("$f"); done
if [ ${#aerial[@]} -gt 0 ]; then gif redrocks_aerial_scan "${aerial[@]}"; fi
