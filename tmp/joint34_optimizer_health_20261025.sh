set -e
cat /proc/loadavg
/data1/shenglin.zeng/sdxl_eval_uniform_20261025/runtime/bin/python - <<'PY'
from pathlib import Path
for pid in [2658575,2658576]:
 p=Path('/proc')/str(pid)
 print('PROCESS',pid,'wchan', (p/'wchan').read_text())
 print('\n'.join(x for x in (p/'status').read_text().splitlines() if x.split(':')[0] in ['State','VmRSS','VmSwap','Threads','voluntary_ctxt_switches','nonvoluntary_ctxt_switches']))
 print('CGROUP',(p/'cgroup').read_text())
p=Path('/sys/fs/cgroup/cpu.max')
if p.exists():print('CPU_MAX',p.read_text())
PY
nvidia-smi --query-gpu=index,pcie.link.gen.current,pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max --format=csv,noheader
