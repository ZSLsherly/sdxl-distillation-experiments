set -e
printf 'CURRENT GPU PIDS\n'
timeout 12 nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
printf 'STOP ONLY THIS WORKFLOW\n'
kill -TERM 2658575 2658576 2658520 2658510 2>/dev/null || true
printf 'SIGNALS_SENT\n'
