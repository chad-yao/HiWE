#!/bin/bash
# Convert demo videos to small GIFs for GitHub README
# Usage: ./scripts/convert_to_gif.sh
# Requires: ffmpeg

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

mkdir -p media/gifs/Task1 media/gifs/Task2 media/gifs/Task3

convert() {
  local inf="$1"
  local outf="${inf/videos/gifs}"
  outf="${outf%.mp4}.gif"
  echo "Converting $inf"
  ffmpeg -y -i "$inf" -t 15 -vf "fps=6,scale=320:-1:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=128[p];[s1][p]paletteuse=dither=bayer" -loop 0 "$outf"
}

for f in media/videos/Task1/*.mp4 media/videos/Task2/*.mp4 media/videos/Task3/*.mp4; do
  [ -f "$f" ] && convert "$f"
done

echo "Done. GIFs in media/gifs/"
