set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
printf 'RUN EXIT\n'
cat logs/smoke_pipeline.exit
python summarize_smoke.py
python -m pip freeze > logs/environment-freeze.txt
git -C src/SparkDiffusion diff > logs/deployment-changes.patch
printf '\nGPU AFTER TRAINING\n'
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
printf '\nDISK USAGE\n'
du -sh envs models cache datasets outputs
