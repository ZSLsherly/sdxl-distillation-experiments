set -eu
if [ -r /proc/2481434/cmdline ]; then
  cmd=$(tr '\0' ' ' < /proc/2481434/cmdline)
  case "$cmd" in
    '/data1/shenglin.zeng/dmd2_official_recipe_20261121/env/bin/python test_cpu.py ') kill -TERM 2481434;;
    *) echo PID_NOT_MATCHED; exit 1;;
  esac
fi
export CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
cd "$R/code"
"$R/env/bin/python" -u - <<'PY'
import faulthandler,runpy,json,hashlib
from pathlib import Path
faulthandler.dump_traceback_later(30,repeat=True)
print('CPU_LOSS_CHECK_START',flush=True)
runpy.run_path('test_cpu.py',run_name='__main__')
faulthandler.cancel_dump_traceback_later()
r=Path('..'); checks={str(p.relative_to(r)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (r/'code').rglob('*.py')}
checks['config.json']=hashlib.sha256((r/'config.json').read_bytes()).hexdigest()
(r/'source_manifest.json').write_text(json.dumps(checks,indent=2))
print('CPU_VALIDATION_AND_MANIFEST_READY',len(checks))
PY
