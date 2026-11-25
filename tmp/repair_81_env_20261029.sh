set -euo pipefail
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
cd "$root"
export LD_LIBRARY_PATH="$root/runtime/lib:${LD_LIBRARY_PATH:-}"
export TMPDIR="$root/tmp" PIP_CACHE_DIR="$root/cache/pip" CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
env/bin/python -m pip --version
mkdir -p tmp/compat_wheels
tar -xf tmp/compat_wheels.tar -C tmp/compat_wheels
env/bin/python -m pip install --disable-pip-version-check --no-index --find-links tmp/compat_wheels --only-binary=:all: --no-deps --force-reinstall numpy==1.26.4 scipy==1.14.1 Pillow==10.4.0
env/bin/python - <<'PY'
import torch,diffusers,transformers,numpy,scipy,PIL
print('VERSIONS',torch.__version__,diffusers.__version__,transformers.__version__,numpy.__version__,scipy.__version__,PIL.__version__,flush=True)
assert not torch.cuda.is_available()
PY
env/bin/python code/sdxl_research/test_turning_batch64_workflow.py
env/bin/python code/sdxl_research/train.py --help >/dev/null
env/bin/python tmp/estimate_81_storage_20261029.py
echo ENVIRONMENT_READY
