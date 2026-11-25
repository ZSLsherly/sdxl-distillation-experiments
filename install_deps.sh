set -eu
ROOT=/data5/shenglin.zeng
export TMPDIR="$ROOT/tmp" PIP_CACHE_DIR="$ROOT/cache/pip" XDG_CACHE_HOME="$ROOT/cache" HF_HOME="$ROOT/cache/huggingface" PYTHONDONTWRITEBYTECODE=1
while kill -0 2973130 2>/dev/null; do sleep 5; done
PY="$ROOT/envs/crossdistill/bin/python"
"$PY" -c 'import torch;print(torch.__version__,torch.version.cuda,torch.cuda.is_available())'
grep -v '^triton' "$ROOT/src/SparkDiffusion/requirements.txt" > "$ROOT/tmp/requirements-a100.txt"
"$PY" -m pip install -r "$ROOT/tmp/requirements-a100.txt" 'transformers<5' 'diffusers<0.36' sentencepiece boto3 termcolor iopath --disable-pip-version-check -i https://pypi.org/simple
"$PY" -m pip check
printf 'DEPENDENCIES_COMPLETE\n'
