set -eu
source /data5/shenglin.zeng/env.sh
BASE=/data4/liushaozhen/miniforge3
OWN="$CROSSDISTILL_ROOT/envs/python-runtime"
mkdir -p "$OWN/bin" "$OWN/lib/python3.12"
cp -a "$BASE/bin/python3.12" "$OWN/bin/"
cp -a "$BASE"/lib/lib*.so* "$OWN/lib/"
tar -C "$BASE/lib/python3.12" --exclude=site-packages --exclude=__pycache__ -cf - . | tar -C "$OWN/lib/python3.12" -xf -
"$OWN/bin/python3.12" -c 'import ssl,sqlite3,ctypes,zlib;print("OWN_PYTHON_OK")'
python - <<'PY'
from pathlib import Path
root=Path('/data5/shenglin.zeng')
p=root/'envs/crossdistill/pyvenv.cfg'
text=p.read_text();text=text.replace('/data4/liushaozhen/miniforge3',str(root/'envs/python-runtime'))
p.write_text(text)
PY
python -c 'import sys,torch,ssl;print(sys.base_prefix,sys.prefix,torch.__version__,torch.cuda.is_available())'
printf '\nRESOLVED CONFIG\n'
grep -E 'use_rola_attn|loss_scale_dmd|loss_scale_tdm|fsdp_shard_size|optim_type|master_weights|capturable|max_iter|rf_t_max' "$CROSSDISTILL_ROOT/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/config.yaml"
tail -5 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
