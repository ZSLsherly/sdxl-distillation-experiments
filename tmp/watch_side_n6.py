from backup_a800_nov5 import lab,run
from pathlib import Path
import time,json,datetime
p=Path('output/sdxl-crossdistill/a800_archive_20261106');p.mkdir(exist_ok=True)
LAB='/data5/shenglin.zeng/checkpoints/a800_archive_20261106'
deadline=time.monotonic()+24*3600
while time.monotonic()<deadline:
 try:
  with lab() as d:
   complete=run(d,'test ! -f '+LAB+'/COMPLETE.json || echo complete')
   if complete:
    print('BACKUP_COMPLETE; performing independent verification and cleanup',flush=True)
    exec(compile(Path('tmp/finish_side_n6.py').read_text(encoding='utf-8'),'tmp/finish_side_n6.py','exec'))
    break
   state=run(d,"du -s -B1 "+LAB+"; ps -eo args | grep -F 'python3 "+LAB+"/.pull6/fz_backup.py' | grep -v grep; tail -n 3 "+LAB+"/.pull6/filezilla.log")
   (p/'progress.txt').write_text(datetime.datetime.now().isoformat()+'\n'+state,encoding='utf-8')
   if 'python3 '+LAB+'/.pull6/fz_backup.py' not in state:
    raise RuntimeError('Transfer stopped before COMPLETE; sources untouched. '+state)
   print(datetime.datetime.now().isoformat(),state.splitlines()[0],flush=True)
 except (OSError,EOFError) as e:
  print('Transient connection error',type(e).__name__,flush=True)
 time.sleep(45)
else:
 raise RuntimeError('Backup deadline reached; source files retained pending verified completion.')
