set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
ls -ld "$root"/* | head -25
for p in bin/python bin/python3.12 env/bin/python runtime/bin/python;do test ! -e "$root/$p" || ls -l "$root/$p";done
python3 - <<'PY'
import tarfile
p='/data3/shenglin.zeng/sdxl_turning_ablation_20261028/transfers/environment.tar.gz'
with tarfile.open(p,'r|gz') as t:
 for i,m in enumerate(t):
  if i>=12:break
  print(m.name,m.linkname)
PY
pgrep -u "$(id -u)" -af '[r]un_81_pipeline_20261029|[s]moke_four_gpu_20261028|[t]urning_batch64_workflow.py' || true
