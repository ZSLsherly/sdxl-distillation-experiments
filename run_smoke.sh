#!/usr/bin/env bash
set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
trap 'code=$?; printf "%s\n" "$code" > "$CROSSDISTILL_ROOT/logs/smoke_pipeline.exit"' EXIT
printf 'PREFLIGHT %s\n' "$(date -Is)"
python -m pip check
python - <<'PY'
import torch,subprocess
from pathlib import Path
root=Path('/data5/shenglin.zeng/models/Wan2.1-T2V-1.3B')
for name in ['diffusion_pytorch_model.safetensors','models_t5_umt5-xxl-enc-bf16.pth','Wan2.1_VAE.pth']:
 p=root/name;assert p.is_file(),p
assert torch.cuda.device_count()==4
processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
assert not processes, 'GPUs have running compute processes; stop before occupying them: '+processes
print('PREFLIGHT_PASSED',flush=True)
PY
printf 'EMBEDDINGS_START %s\n' "$(date -Is)"
PREP_STAGE=embeddings torchrun --standalone --nproc_per_node=1 smoke_data.py > logs/prepare_embeddings.log 2>&1
printf 'TEACHER_SAMPLING_START %s\n' "$(date -Is)"
PREP_STAGE=latents torchrun --standalone --nproc_per_node=4 smoke_data.py > logs/prepare_latents.log 2>&1
printf 'TRAINING_START %s\n' "$(date -Is)"
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw --format=csv -l 1 > logs/training_gpu_samples.csv &
MONITOR_PID=$!
trap 'code=$?; kill "$MONITOR_PID" 2>/dev/null || true; printf "%s\n" "$code" > "$CROSSDISTILL_ROOT/logs/smoke_pipeline.exit"' EXIT
torchrun --standalone --nproc_per_node=4 -m scripts.train --config=smoke_config.py > logs/train_smoke.log 2>&1
printf 'TRAINING_COMPLETE %s\n' "$(date -Is)"
