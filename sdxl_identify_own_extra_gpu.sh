source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
p=Path('/proc/3051943/cmdline')
if p.exists():
 args=p.read_bytes().replace(b'\0',b' ').decode(errors='replace')
 own=any(s in args for s in ['sdxl_teacher_turning.py','sdxl_summarize_turning.py','sdxl_stage_metrics.py','sdxl_pilot_evaluate.py','sdxl_preview_pilot.py'])
 print('EXTRA_GPU_PROCESS_IS_THIS_TASK',own)
 if own:print('TASK_COMMAND',args)
PY
