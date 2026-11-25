tail -70 /data3/shenglin.zeng/clip_missing_20261113/arms/A_pcm10_patchgan/worker.log
cat /data3/shenglin.zeng/clip_missing_20261113/failure_gpu5.json
nvidia-smi -i 5 --query-compute-apps=pid,used_memory --format=csv,noheader
