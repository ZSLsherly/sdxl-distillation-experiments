set -eu
ROOT=/data5/shenglin.zeng
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" XDG_CACHE_HOME="$ROOT/cache" HF_HOME="$ROOT/cache/huggingface" PYTHONDONTWRITEBYTECODE=1
if [ ! -x "$ROOT/envs/crossdistill/bin/python" ]; then /data4/liushaozhen/miniforge3/bin/python -m venv --copies "$ROOT/envs/crossdistill"; fi
"$ROOT/envs/crossdistill/bin/python" -m pip install --disable-pip-version-check torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 > "$ROOT/logs/install_torch.log" 2>&1 &
printf 'TORCH_INSTALL_PID=%s\n' "$!"
