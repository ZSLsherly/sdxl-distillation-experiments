source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cd "$root"
nohup bash -c '
root=/data5/shenglin.zeng/sdxl_crossdistill
for i in $(seq 1 1440);do
 if grep -q PILOT_TRAINING_COMPLETE logs/hparam_pipeline.log;then
  if test -f models/metrics/clip_manifest.json;then
   export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
   python src/sdxl_pilot_evaluate.py
   status=$?
   echo "$status" > logs/pilot_assessment.exit
   exit "$status"
  fi
 fi
 if ! kill -0 "$(cat logs/hparam_pipeline.pid)" 2>/dev/null && ! grep -q PILOT_TRAINING_COMPLETE logs/hparam_pipeline.log;then echo TRAINING_PIPELINE_FAILED;echo 1 > logs/pilot_assessment.exit;exit 1;fi
 sleep 10
done
echo 124 > logs/pilot_assessment.exit
' > logs/pilot_assessment.log 2>&1 < /dev/null &
echo $! > logs/pilot_assessment.pid
cat logs/pilot_assessment.pid
python - <<'PY'
import json,statistics
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for p in sorted((root/'outputs/hparam_pilot').glob('lr*')):
 q=p/'rank0.jsonl'
 if not q.exists():continue
 rows=[json.loads(s) for s in q.read_text().splitlines()]
 s=[r for r in rows if r['phase']=='student']
 print(p.name,'updates',len(rows),'student',len(s),'last',rows[-1] if rows else None)
 if s:print('STUDENT_GRAD_NORM',min(r['grad_norm'] for r in s),max(r['grad_norm'] for r in s),'CLIPPING',sum(r['clip_scale']<1 for r in s))
print('CLIP_DOWNLOAD', (root/'logs/clip_download.log').read_text()[-500:])
PY
