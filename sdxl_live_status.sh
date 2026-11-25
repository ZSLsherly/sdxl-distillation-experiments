source /data5/shenglin.zeng/env.sh
root="$CROSSDISTILL_ROOT/sdxl_crossdistill"
tail -n 18 "$root/logs/train.log"
tail -n 5 "$root/logs/baselines_download.log"
tail -n 5 "$root/logs/finish.log"
if [ -f "$root/logs/comparison.log" ]; then tail -n 12 "$root/logs/comparison.log";fi
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
if [ -f "$root/logs/train.exit" ]; then echo TRAIN_EXIT;cat "$root/logs/train.exit";fi
if [ -f "$root/logs/comparison.exit" ]; then echo COMPARISON_EXIT;cat "$root/logs/comparison.exit";fi
