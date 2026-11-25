set -eu
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
find "$p/models/sdxl-base-1.0" -maxdepth 2 -type f -printf '%P %s\n' | head -100
find "$p/models/metrics/clean_fid" -maxdepth 3 -type f -printf '%P %s\n'
ls -l "$p/models/sdxl-base-1.0"; du -sh "$p/reference_features/real"
