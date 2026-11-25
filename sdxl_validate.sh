set -eu
source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
python -m py_compile "$root/src/sdxl_objectives.py" "$root/src/sdxl_train.py" "$root/src/sdxl_prepare.py"
python "$root/src/sdxl_objectives.py"
tail -n 6 "$root/logs/download.log"
tail -n 6 "$root/logs/data_download.log"
