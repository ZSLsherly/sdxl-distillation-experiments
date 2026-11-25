B=/data1/shenglin.zeng/dymd_prior_8_20261101
python3 - <<'PY'
from pathlib import Path
b=Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
p=b/'dependency_resolution.log'
print('\n'.join(p.read_text(errors='replace').replace('\r','\n').splitlines()[-12:]))
PY
ps -u shenglin.zeng -o pid,ppid,etime,args | grep -E 'pip install|download_parallel|conda create' | head -15
