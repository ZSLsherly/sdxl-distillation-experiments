export CROSSDISTILL_ROOT=/data5/shenglin.zeng
export PATH="$CROSSDISTILL_ROOT/envs/crossdistill/bin:/usr/local/cuda/bin:$PATH"
export TMPDIR="$CROSSDISTILL_ROOT/tmp"
export TMP="$TMPDIR" TEMP="$TMPDIR"
export XDG_CACHE_HOME="$CROSSDISTILL_ROOT/cache"
export HF_HOME="$CROSSDISTILL_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TORCH_HOME="$CROSSDISTILL_ROOT/cache/torch"
export TRITON_CACHE_DIR="$CROSSDISTILL_ROOT/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$CROSSDISTILL_ROOT/cache/torchinductor"
export CUDA_CACHE_PATH="$CROSSDISTILL_ROOT/cache/cuda"
export PIP_CACHE_DIR="$CROSSDISTILL_ROOT/cache/pip"
export CONDA_PKGS_DIRS="$CROSSDISTILL_ROOT/cache/conda/pkgs"
export WANDB_DIR="$CROSSDISTILL_ROOT/logs/wandb"
export WANDB_CACHE_DIR="$CROSSDISTILL_ROOT/cache/wandb"
export WANDB_CONFIG_DIR="$CROSSDISTILL_ROOT/cache/wandb-config"
export WANDB_MODE=offline
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$CROSSDISTILL_ROOT/src/SparkDiffusion:$CROSSDISTILL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$CROSSDISTILL_ROOT/envs/crossdistill/lib/python3.12/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4
export IMAGINAIRE_OUTPUT_ROOT="$CROSSDISTILL_ROOT/outputs"
