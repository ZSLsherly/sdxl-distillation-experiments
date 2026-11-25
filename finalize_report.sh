set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python summarize_smoke.py > logs/smoke_summary.log
test -s outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/checkpoints/iter_000000010/exported.pth
printf 'GPU COMPUTE PROCESSES\n'
nvidia-smi --query-compute-apps=pid,process_name --format=csv
printf '\nOWN TRAINING PROCESSES\n'
ps -u "$(id -un)" -o pid,args | grep -F '/data5/shenglin.zeng/envs/crossdistill/bin/python -u -m scripts.train' | grep -v grep || true
printf '\nREPORT_READY\n'
