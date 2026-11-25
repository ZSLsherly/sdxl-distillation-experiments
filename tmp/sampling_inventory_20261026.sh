set -eu
id -un
for p in /data1/shenglin.zeng "$HOME/miniconda3/envs" "$HOME/anaconda3/envs"; do
 if test -d "$p"; then find "$p" -maxdepth 4 -type d \( -name '*sd3*' -o -name '*sdxl*' -o -name 'site-packages' -o -name 'runtime' -o -name 'env' -o -name 'models' \) | head -40; fi
done
ls -ld /data1 /data1/shenglin.zeng 2>/dev/null || true
python3 - <<'PY'
import sys,importlib.util
print(sys.version)
print({k:bool(importlib.util.find_spec(k)) for k in ['torch','diffusers','transformers']})
PY
