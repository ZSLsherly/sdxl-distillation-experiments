set -eu
R=/data/shenglin.zeng/a50k_20261123
PY=/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python
"$PY" -m py_compile "$R/code/prepare_queue.py"
if pgrep -f '^/data3/shenglin.zeng/sdxl_turning_ablation_20261028/env/bin/python -u /data/shenglin.zeng/a50k_20261123/code/prepare_queue.py' >/dev/null; then
 echo EXISTING_PREPARE_QUEUE
else
 nohup "$PY" -u "$R/code/prepare_queue.py" > "$R/records/prepare_queue.log" 2>&1 < /dev/null &
 echo "PREPARE_QUEUE_PID=$!"
fi
sleep 2
cat "$R/records/prepare_status.json"
sha256sum "$R/datasets/sources/vae_latents_004.pt"
