set -eu
. /root/sdxl_research/env.sh
cd "$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
export PYTHONPATH="$PWD"
export CUDA_VISIBLE_DEVICES=''
python - <<'PY'
from pathlib import Path
import hashlib,json,os
import control_after_bc as a
m=json.loads((a.evalout/'manifest.json').read_text())
o=a.prepare_control()
c=json.loads((o/'manifest.json').read_text())
assert {k:v for k,v in c.items() if k!='models'} == {k:v for k,v in m.items() if k!='models'}
assert list(c['models'])==['A'] and not (a.job/'late_mix').exists()
ep=Path('evaluate.py');qp=Path('run_bc_coco_then_winner.py')
old=ep.read_text()
q=qp.read_text()
mark='    else:core.metrics()'
replacement="""    else:
        core.metrics()
        from control_after_bc import evalout, run_control
        if core.OUT == evalout.resolve():
            run_control()"""
qmark="    cfgfile=root/'code/sdxl_research/configs'/f'turning_{winner}_quota_a800_g160.json'"
qnew="    from control_after_bc import run_control\n    run_control()\n"+qmark
for path,src,needle,new in [(qp,q,qmark,qnew),(ep,old,mark,replacement)]:
    if new in src: continue
    assert src.count(needle)==1
    result=src.replace(needle,new)
    compile(result,str(path),'exec')
    stage=path.with_suffix('.next')
    stage.write_text(result);os.replace(stage,path)
assert replacement in ep.read_text() and qnew in qp.read_text()
print('VERIFIED: active B/C metrics will await A; restart gate present; identical 10000 evaluation records and 128 visual indices.')
print('A checkpoint:',c['models']['A']['checkpoint'])
PY
tail -2 "$SDXL_WORK/logs/bc_coco_winner_queue.log"
cat "$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/coco10k/status_0.json"
