set -eu
. /root/sdxl_research/env.sh
CUDA_VISIBLE_DEVICES='' python - <<'PY'
import math,os,json,statistics,tarfile
from pathlib import Path
w=Path(os.environ['SDXL_WORK']); d=Path('/root/autodl-tmp/h20_gan_mb16/code_fast')
out=w/'outputs/h20_gan_mb16_fast'
b=json.loads((out/'benchmark.json').read_text()); rows=[json.loads(x) for x in (out/'rank0.jsonl').read_text().splitlines()]
gs=[r for r in rows if r['phase']=='student_done']
fs=[r for r in rows if r['phase'] in ('fake','fake_D')]
probes=[r for r in rows if r['phase']=='gan_backward_probe']
assert [r['g'] for r in gs]==[1,2,3] and len(fs)==6 and sum(r['phase']=='fake_D' for r in fs)==3
assert len(probes)==3 and all(r['weighted_accumulated_latent_rms']>0 and math.isfinite(r['weighted_accumulated_latent_rms']) for r in probes)
assert all(math.isfinite(r[k]) for r in gs for k in ('grad_norm','pcm_loss','terminal_dm_loss','gan_loss'))
assert all([fs[2*i]['fake_stage'],fs[2*i+1]['fake_stage']]==[g['stage'],4] for i,g in enumerate(gs))
assert b['status']=='complete' and b['accumulation']==4 and b['microbatch_per_gpu']==16 and b['effective_batch']==64
transfers=[r for r in rows if r['phase']=='fake_adam_transfer']; gpu=max(int(l.split(',')[1]) for l in (d/'gpu.csv').read_text().splitlines() if len(l.split(','))==3)
cycles=b['cycle_seconds']; mean=statistics.mean(cycles[1:]);transfer=sum(r['seconds'] for r in transfers if r['g'] in (2,3))/2
v=dict(cycles=cycles,steady_mean_seconds=mean,student_samples_per_s=64/mean,peak_allocated_GiB=b['peak_allocated_GiB'],peak_reserved_GiB=b['peak_reserved_GiB'],device_sampled_peak_GiB=gpu/1024,fake_adam_copy_seconds_per_G=transfer,estimate_G160_hours=160*mean/3600,loop_offload='student Adam between updates; fake Adam moments during student phase',scope='3G/6fake/3D, no inference quality evaluation')
(d/'measurement.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
with tarfile.open(w/'tmp/h20_mb16_results.tar.gz','w:gz') as a:
 for n in ('measurement.json','gpu.csv','h20.log','config.json','sam_pixel.py','train.py','run_h20_16.py'):a.add(d/n,arcname=n)
 for n in ('benchmark.json','rank0.jsonl','status.json'):a.add(out/n,arcname=n)
PY
