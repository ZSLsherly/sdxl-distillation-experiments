set -eu
ROOT=/data5/shenglin.zeng
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" XDG_CACHE_HOME="$ROOT/cache" HF_HOME="$ROOT/cache/huggingface" PYTHONDONTWRITEBYTECODE=1
cat > "$ROOT/download_models.py" <<'PY'
import os
from huggingface_hub import snapshot_download
snapshot_download('Wan-AI/Wan2.1-T2V-1.3B',local_dir='/data5/shenglin.zeng/models/Wan2.1-T2V-1.3B',allow_patterns=['*.pth','*.safetensors','config.json','google/*','LICENSE.txt'],max_workers=3)
print('TEACHER_DOWNLOAD_COMPLETE',flush=True)
PY
nohup env PYTHONPATH="$ROOT/envs/download-tools" HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 HF_HUB_DOWNLOAD_TIMEOUT=120 /data4/liushaozhen/miniforge3/bin/python "$ROOT/download_models.py" > "$ROOT/logs/download_models.log" 2>&1 < /dev/null &
printf 'MODEL_DOWNLOAD_PID=%s\n' "$!"
printf '\nDOWNLOAD TOOLS\n'
tail -3 "$ROOT/logs/download-tools.log"
printf '\nREPOSITORY STATUS\n'
ls "$ROOT/src/SparkDiffusion"
