set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_4gpu_20261123
mkdir -p "$R/logs" "$R/tmp" "$R/cache"
if [ ! -e "$R/env" ]; then ln -s /data1/shenglin.zeng/dmd2_matched_10k_20261119/env "$R/env"; fi
df -h /data1
