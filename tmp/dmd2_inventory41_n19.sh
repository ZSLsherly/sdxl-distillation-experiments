set -eu
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/env/bin/python
ls -l "$p"
CUDA_VISIBLE_DEVICES='' "$p" - <<'PY'
import torch,diffusers,transformers,accelerate
print('versions',torch.__version__,diffusers.__version__,transformers.__version__,accelerate.__version__)
import importlib.util
for n in ['wandb','lmdb','peft','matplotlib']:print(n,importlib.util.find_spec(n) is not None)
PY
find /data1/shenglin.zeng -maxdepth 5 -name prepared.pt -o -name model_index.json -o -name rclone
nvidia-smi topo -m
