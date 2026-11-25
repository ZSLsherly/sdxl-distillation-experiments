set -eu
r=/data5/shenglin.zeng
ls "$r/tools"
ls "$r/envs"
ls "$r/models"
ls "$r/datasets"
ls "$r/sdxl_crossdistill/datasets" 2>/dev/null || true
find "$r" -maxdepth 4 -name rclone -o -name env.sh -o -name prepared.pt -o -name model_index.json
