import ast
from pathlib import Path
p=Path('tmp/run_shared_turning_probe_3090.py');s=p.read_text(encoding='utf-8')
s=s.replace('GPU=0',"GPU=0\nPAUSE=OUT/'YIELD_RETRY1'\nPAUSE.unlink(missing_ok=True)")
s=s.replace("'TMPDIR':str(ROOT/'tmp')}","'TMPDIR':str(ROOT/'tmp'),'SDXL_SHARED_YIELD_FILE':str(PAUSE)}")
s=s.replace('short_{arm}','short_{arm}_retry1')
s=s.replace('start=time.time();print','start=time.time();calm=0;print')
s=s.replace('if new or grown or busy or free<4000',
'''if busy:
      calm=0
      if not PAUSE.exists():print('YIELD_TO_OTHER',busy,flush=True)
      PAUSE.touch()
     else:
      calm+=1
      if calm>=3 and PAUSE.exists():PAUSE.unlink();print('RESUME_PROBE',flush=True)
     if new or grown or free<4000''')
s=s.replace('   finally:\n','   finally:\n    PAUSE.unlink(missing_ok=True)\n')
s=s.replace('paired_GPU_probe_complete.json','paired_GPU_probe_retry1_complete.json').replace('paired_GPU_probe_error.json','paired_GPU_probe_retry1_error.json')
ast.parse(s);Path('tmp/run_shared_turning_probe_3090_retry1.py').write_text(s,encoding='utf-8')
ast.parse(Path('sdxl_research/train.py').read_text(encoding='utf-8'))
print('Retry guard and training entry parsed')
