set -eu
r=/data5/shenglin.zeng/sdxl_crossdistill/models/sdxl-base-1.0
find "$r" -maxdepth 2 -type f -printf '%P %s\n' | head -35
