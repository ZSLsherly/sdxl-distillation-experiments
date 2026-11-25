source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json,statistics,math
from pathlib import Path
root=Path('outputs/gan_stage_pilot_v2');result={}
for name in ['dmd','gan4','joint4','joint34_matched']:
    p=root/name;f=p/'rank0.jsonl'
    rows=[json.loads(x) for x in f.read_text().splitlines() if x.strip()] if f.exists() else []
    gs=[r for r in rows if r['phase']=='student'];fs=[r for r in rows if r['phase']=='fake_score']
    s=json.loads((p/'summary.json').read_text()) if (p/'summary.json').exists() else None
    result[name]={'status':s['status'] if s else ('running' if p.exists() else 'queued'),'student_updates':len(gs),'fake_updates':len(fs),'gan_G_updates':sum(r['gan_active'] for r in gs),'logged_train_minutes':sum(r['seconds'] for r in rows)/60,'finite':all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows),'student_seconds_median':statistics.median(r['seconds'] for r in gs) if gs else None,'fake_seconds_median':statistics.median(r['seconds'] for r in fs) if fs else None,'peak_allocated_GiB':max((r['peak_allocated_GiB'] for r in rows),default=0),'clipped_G_updates':sum(r['clip_scale']<1 for r in gs)}
    if s:result[name].update(train_minutes=s['train_seconds']/60,own_peak_MiB=s.get('nvidia_smi_peak_own_process_MiB_per_gpu'),checkpoint_shards=sum((p/f'student_rank{r}.pt').exists() for r in range(4)))
    if (p/'validation_after_rank0.json').exists():
        vals=[]
        for r in range(4):vals+=json.loads((p/f'validation_after_rank{r}.json').read_text())
        result[name]['validation_finite']=all(x['finite'] for x in vals)
print(json.dumps(result,indent=2))
print('ASSESSMENT_READY',(root/'assessment/result.json').exists())
PY
