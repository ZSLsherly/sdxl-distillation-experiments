set -euo pipefail
root=/root/autodl-tmp/sdxl_research
cd "$root"
exec 9>setup.lock
flock -n 9
trap 'echo SETUP_FAILED_LINE=$LINENO; echo failed > setup.status' ERR
echo waiting_for_assets > setup.status
for n in $(seq 1 4320); do
 test ! -f transfers/manifest.json || break
 sleep 10
done
test -f transfers/manifest.json
echo verifying > setup.status
/root/miniconda3/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('transfers')
for name,e in json.loads((p/'manifest.json').read_text()).items():
 h=hashlib.sha256()
 with (p/name).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 assert (p/name).stat().st_size==e['bytes'] and h.hexdigest()==e['sha256'],name
 print('VERIFIED',name,flush=True)
PY
echo extracting > setup.status
tar -xzf transfers/environment.tar.gz -C "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
runtime/bin/python3.12 -m venv --copies --without-pip env
tar -xf transfers/sdxl_sampling.tar -C "$root"
mkdir -p datasets/parallel_10k_seed20261018 cache tmp
mv transfers/prepared.pt datasets/parallel_10k_seed20261018/prepared.pt
mv transfers/dataset_manifest.json datasets/parallel_10k_seed20261018/manifest.json
tar -xzf transfers/code.tar.gz -C code
cat > env.sh <<'ENV'
export SDXL_ROOT=/root/autodl-tmp/sdxl_research
export PATH="$SDXL_ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$SDXL_ROOT/runtime/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$SDXL_ROOT/cache"
export TMPDIR="$SDXL_ROOT/tmp"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
ENV
source env.sh
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$root/code/sdxl_research"
echo validating > setup.status
python -m unittest test_boundary_global_noise test_boundary test_turning_batch64_workflow
python code/sdxl_research/train.py --help > logs/train_help.txt
python - <<'PY'
import json,torch,importlib.metadata as m
from pathlib import Path
from diffusers import UNet2DConditionModel,AutoencoderKL,DDIMScheduler
root=Path.cwd()
d=torch.load(root/'datasets/parallel_10k_seed20261018/prepared.pt',weights_only=True,mmap=True,map_location='cpu')
assert d['train_count']==10000
assert tuple(d['latents'].shape[1:])==(4,128,128)
base=root/'models/sdxl-base-1.0'
u=UNet2DConditionModel.from_pretrained(base/'unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
count=sum(p.numel() for p in u.parameters());del u
v=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True);del v
DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
result=dict(status='cpu_preparation_complete',train_count=10000,latent_shape=list(d['latents'].shape),unet_parameters=count,cuda_build=torch.version.cuda,gpu_available=torch.cuda.is_available(),packages={p:m.version(p) for p in ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','safetensors','Pillow']},gpu_validation='not_run',training_started=False)
(root/'preparation_complete.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
PY
echo complete > setup.status
df -h "$root"
