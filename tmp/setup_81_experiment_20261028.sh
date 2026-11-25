set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$root/tmp" CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
python3 - <<'PY'
import hashlib,json
from pathlib import Path
p=Path('transfers'); data=json.loads((p/'manifest.json').read_text())
for name, expected in data.items():
    h=hashlib.sha256(); path=p/name
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    assert path.stat().st_size==expected['bytes'] and h.hexdigest()==expected['sha256'], name
    print('VERIFIED',name,flush=True)
PY
nice -n 19 ionice -c 2 -n 7 tar -xzf transfers/environment.tar.gz -C "$root"
runtime/bin/python -m venv --copies --without-pip env
nice -n 19 ionice -c 2 -n 7 tar -xf transfers/sdxl_sampling.tar -C "$root"
mkdir -p datasets/parallel_10k_seed20261018 code
test ! -e datasets/parallel_10k_seed20261018/prepared.pt
mv transfers/prepared.pt datasets/parallel_10k_seed20261018/prepared.pt
mv transfers/dataset_manifest.json datasets/parallel_10k_seed20261018/manifest.json
tar -xzf tmp/turning_code.tar.gz -C code
env/bin/python - <<'PY'
import torch,diffusers,transformers
print('VERSIONS',torch.__version__,diffusers.__version__,transformers.__version__)
assert not torch.cuda.is_available()
PY
env/bin/python code/sdxl_research/test_turning_batch64_workflow.py
env/bin/python code/sdxl_research/train.py --help >/dev/null
df -h "$root" | tail -1
echo ENVIRONMENT_READY
