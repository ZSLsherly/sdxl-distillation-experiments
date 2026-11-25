set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
python "$root/src/sdxl_summarize.py"
du -sh "$root/outputs/smoke_1024"
