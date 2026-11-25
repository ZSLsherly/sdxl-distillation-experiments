date -Is
python3 - <<'PY'
import time,os
from pathlib import Path
p=Path('/proc/33807')
if p.exists():
 def stat():
  a=(p/'stat').read_text().split();return int(a[13])+int(a[14])
 before=stat();time.sleep(3);after=stat()
 print('CPU_seconds_over_3s',(after-before)/os.sysconf('SC_CLK_TCK'))
 print('WCHAN',(p/'wchan').read_text())
 print('\n'.join(x for x in (p/'status').read_text().splitlines() if x.startswith(('State','VmRSS','VmSwap','Threads'))))
 print('LOADAVG',Path('/proc/loadavg').read_text())
PY
nvidia-smi -i 0 --query-gpu=utilization.gpu,utilization.memory,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader
tail -10 /data1/shenglin.zeng/teacher_turning_20261026/outputs/turning_boundary_ablation_20261027/short_normalized_u_retry1.log
