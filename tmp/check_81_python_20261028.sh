set -eu
export CUDA_VISIBLE_DEVICES=''
for py in /data1/shaozhen.liu/miniconda3/bin/python /usr/bin/python3; do
 if test -x "$py"; then
  echo PYTHON="$py"
  "$py" -c 'import importlib.util; print({k: bool(importlib.util.find_spec(k)) for k in ("torch","diffusers","transformers","accelerate","safetensors","numpy","PIL")})' || true
 fi
done
ls -ld /data3
