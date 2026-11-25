set -euo pipefail
src=/root/autodl-tmp/sdxl_research
dst=/root/sdxl_research
work=/root/autodl-tmp/sdxl_runs
exec 9>"$src/tmp/system_migration.lock"
flock -n 9
test ! -e "$dst"
/root/miniconda3/bin/python - <<'PY'
import shutil,subprocess
from pathlib import Path
src=Path('/root/autodl-tmp/sdxl_research')
names=['env','runtime','models','datasets','code','conditions.pt','prompts.json','preparation_complete.json']
size=sum(int(subprocess.check_output(['du','-sb',str(src/n)]).split()[0]) for n in names)
free=shutil.disk_usage('/root').free
print('COPY_BYTES',size,'SYSTEM_FREE_BYTES',free,flush=True)
assert free > size+3*1024**3,'Keep at least 3GiB free on system disk'
PY
mkdir -p "$dst" "$work"/{outputs,logs,cache,tmp}
for name in env runtime models datasets code conditions.pt prompts.json preparation_complete.json; do
 rsync -a "$src/$name" "$dst/"
done
printf 'COPIED\n'
# Compare file contents before changing relocated interpreter paths.
for name in env runtime models datasets code conditions.pt prompts.json preparation_complete.json; do
 difference=$(rsync -rcni "$src/$name" "$dst/")
 test -z "$difference" || { printf '%s\n' "$difference"; exit 1; }
done
printf 'COPY_CHECKSUMS_OK\n'
export LD_LIBRARY_PATH="$dst/runtime/lib:${LD_LIBRARY_PATH:-}"
"$dst/runtime/bin/python3.12" -m venv --copies --without-pip "$dst/env"
/root/miniconda3/bin/python - <<'PY'
from pathlib import Path
old='/root/autodl-tmp/sdxl_research';new='/root/sdxl_research'
for p in (Path(new)/'env/bin').iterdir():
 if p.is_file() and not p.is_symlink():
  try:s=p.read_text()
  except (UnicodeError,OSError):continue
  if old in s:p.write_text(s.replace(old,new))
p=Path(new)/'code/sdxl_research/train.py'
s=p.read_text().replace('default=Path("/data5/shenglin.zeng/sdxl_crossdistill")','default=Path(os.environ.get("SDXL_ROOT", "/data5/shenglin.zeng/sdxl_crossdistill"))');p.write_text(s)
p=Path(new)/'code/sdxl_research/turning_batch64_workflow.py'
p.write_text(p.read_text().replace('shutil.disk_usage(root).free','shutil.disk_usage(out).free'))
PY
for name in outputs logs cache tmp; do ln -s "$work/$name" "$dst/$name"; done
cat > "$dst/env.sh" <<'ENV'
export SDXL_ROOT=/root/sdxl_research
export SDXL_WORK=/root/autodl-tmp/sdxl_runs
export PATH="$SDXL_ROOT/env/bin:$PATH"
export LD_LIBRARY_PATH="$SDXL_ROOT/runtime/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="$SDXL_WORK/cache"
export XDG_CACHE_HOME="$SDXL_WORK/cache"
export TORCH_HOME="$SDXL_WORK/cache/torch"
export TMPDIR="$SDXL_WORK/tmp"
export TORCHINDUCTOR_CACHE_DIR="$SDXL_WORK/cache/inductor"
export TRITON_CACHE_DIR="$SDXL_WORK/cache/triton"
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$SDXL_WORK"/{outputs,logs,cache,tmp}
ENV
cat > "$dst/launch.sh" <<'SH'
#!/bin/bash
set -euo pipefail
source /root/sdxl_research/env.sh
exec "$SDXL_ROOT/env/bin/python" "$SDXL_ROOT/code/sdxl_research/launch.py" --root "$SDXL_ROOT" "$@"
SH
chmod +x "$dst/launch.sh"
# Compatibility entry point for the old activation path.
cp "$src/env.sh" "$src/env.before_system_migration.sh"
printf 'source /root/sdxl_research/env.sh\n' > "$src/env.sh"
source "$dst/env.sh"
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
export CUDA_VISIBLE_DEVICES='' HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python -m unittest test_boundary_global_noise test_boundary test_turning_batch64_workflow
python "$SDXL_ROOT/code/sdxl_research/train.py" --help > "$SDXL_WORK/logs/train_help.txt"
python - <<'PY'
import json,os,sys,shutil
from pathlib import Path
import torch,diffusers,transformers
root=Path(os.environ['SDXL_ROOT']);work=Path(os.environ['SDXL_WORK'])
assert Path(sys.prefix)==root/'env'
assert Path(sys.base_prefix)==root/'runtime'
for p in ['models','datasets','env','runtime','code']:assert not (root/p).is_symlink()
for p in ['outputs','logs','cache','tmp']:assert (root/p).resolve()==work/p
assert os.stat(root/'models').st_dev==os.stat('/root').st_dev
assert os.stat(root/'outputs').st_dev==os.stat('/root/autodl-tmp').st_dev
print('INTERPRETER',sys.executable,sys.prefix,sys.base_prefix)
print('PACKAGES',torch.__version__,diffusers.__version__,transformers.__version__)
result=dict(status='system_migration_cpu_verified',root=str(root),work=str(work),cpu_tests=22,copy_checksum_verified=True,training_started=False,gpu_validation='pending',system_free_GiB=shutil.disk_usage(root).free/1024**3,data_free_GiB=shutil.disk_usage(work).free/1024**3)
(root/'storage_layout.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
df -h / /root/autodl-tmp
