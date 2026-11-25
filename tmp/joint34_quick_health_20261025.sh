set -e
printf 'LOAD\n'
cat /proc/loadavg
printf 'CPU COUNT\n'
getconf _NPROCESSORS_ONLN
printf 'LATEST OWN LOG\n'
tail -n 1 /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/joint34_uniform_10k_2gpu_20261025/resource_monitor.jsonl
printf 'OWN STUDENT STATUS\n'
cat /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/joint34_uniform_10k_2gpu_20261025/phase1/status.json
printf '\n'
