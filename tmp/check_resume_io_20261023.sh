source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
from pathlib import Path
import subprocess
p=Path('outputs/joint4_uniform_10k_2gpu_mb16_20261023/train')
pids=[(p/f'pid_rank{r}').read_text().strip() for r in (0,1)]
print(subprocess.check_output(['ps','-p',','.join(pids),'-o','pid,etime,pcpu,rss,stat,wchan:32'],text=True))
for pid in pids:
 print('IO',pid,Path('/proc',pid,'io').read_text())
 for fd in Path('/proc',pid,'fd').iterdir():
  try:
   target=str(fd.resolve())
   if '/training_state/' in target:print('CHECKPOINT_READING',pid,Path(target).name)
  except OSError:pass
print(subprocess.check_output(['free','-h'],text=True))
PY
