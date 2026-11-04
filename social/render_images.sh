#!/bin/bash
# Render docs/*.svg to PNGs for social media (headless Chrome, 2x). Usage: social/render_images.sh
cd "$(dirname "$0")/.."; C="${CHROME:-/Applications/Google Chrome.app/Contents/MacOS/Google Chrome}"
for n in banner arch life-of-a-matvec adder-tree chip-path packing hardware bug-museum datapath token-timeline gdn-state; do
  vb=$(grep -o 'viewBox="[^"]*"' docs/$n.svg | head -1 | sed 's/viewBox="//;s/"//'); set -- $vb; w=${3%.*}; h=${4%.*}
  "$C" --headless=new --hide-scrollbars --force-device-scale-factor=2 --window-size=$w,$h --screenshot=social/images/$n.png "file://$PWD/docs/$n.svg" >/dev/null 2>&1
  echo "$n ${w}x${h}"
done
