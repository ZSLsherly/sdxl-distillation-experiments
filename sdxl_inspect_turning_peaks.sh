source /data5/shenglin.zeng/env.sh
python - <<'PY'
import json,numpy as np
from pathlib import Path
p=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot/teacher_turning/result.json');r=json.loads(p.read_text())
for name,v in r['matched_controls'].items():print(name,{k:x for k,x in v.items() if not k.endswith('curve')})
cur=r['modes']['base_dt10_bf16']['curves'];t=np.array(r['native_t_grid'])
for step in [999,979,949,919,899,849,749,599,449]:
 i=abs(t-step).argmin();print('AT_T',step,'NEAREST',round(t[i],1),'RATE',cur['rf_full']['smoothed_mean'][i],'COARSE',cur['rf_coarse_centered']['smoothed_mean'][i])
print('TOP_RAW_ANGLE_BINS',sorted(r['native_t_bins20'],key=lambda x:x['mean_angle_degrees'],reverse=True)[:8])
print('TOP_RATE_BINS',sorted(r['native_t_bins20'],key=lambda x:x['mean_rate'],reverse=True)[:8])
PY
