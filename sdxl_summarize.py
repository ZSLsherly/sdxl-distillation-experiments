"""Aggregate measured resource results without claiming quality convergence."""
import json,statistics,hashlib
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs/smoke_1024'
summaries=[json.loads(p.read_text()) for p in sorted(out.glob('summary_rank*.json'))]
assert len(summaries)==4
records=[]
for rank in range(4):
 records.append([json.loads(s) for s in (out/f'rank{rank}.jsonl').read_text().splitlines()])
assert all(len(r)==20 for r in records)
steps=[]
for iteration in range(20):
 rows=[r[iteration] for r in records];assert len({r['phase'] for r in rows})==1
 steps.append({'iteration':iteration,'phase':rows[0]['phase'],'seconds_slowest_rank':max(r['seconds'] for r in rows),'seconds_per_rank':[r['seconds'] for r in rows]})
timing={}
for phase in ['fake_score','student']:
 rows=[r for r in steps if r['phase']==phase];steady=rows[1:]
 timing[phase]={'updates':len(rows),'mean_seconds':statistics.mean(r['seconds_slowest_rank'] for r in rows),'steady_mean_seconds_excluding_first':statistics.mean(r['seconds_slowest_rank'] for r in steady),'min_seconds':min(r['seconds_slowest_rank'] for r in rows),'max_seconds':max(r['seconds_slowest_rank'] for r in rows)}
train_time=max(s['train_seconds'] for s in summaries)
overall=json.loads((out/'summary.json').read_text())
result={'status':'completed_resource_smoke_test','method':'CrossDistill adapted to SDXL native VP/DDIM','training_data':json.loads((root/'datasets/prepared_manifest.json').read_text()),'configuration':overall['config'],'ranks':summaries,'training_time_seconds':train_time,'aggregate_examples_per_second':20*4/train_time,'phase_timing':timing,'steps':steps,'peak_allocated_GiB_per_rank':[s['peak_allocated_GiB'] for s in summaries],'peak_reserved_GiB_per_rank':[s['peak_reserved_GiB'] for s in summaries],'nvidia_smi_peak_own_process_MiB_per_gpu':overall.get('nvidia_smi_peak_own_process_MiB_per_gpu'),'nvidia_smi_peak_total_MiB_per_gpu':overall.get('nvidia_smi_peak_total_MiB_per_gpu'),'quality_claim':'None: four student updates are only sufficient to verify execution, gradients and resource usage','caveat':'GPU1 also had another user process (~1.2GiB, active); its process was left untouched'}
comparison=out/'comparison';paths=sorted(comparison.glob('timing_rank*.json'))
if len(paths)==4:
 values=[row for p in paths for row in json.loads(p.read_text())['records']]
 methods={}
 for name in sorted({r['method'] for r in values}):
  rows=[r for r in values if r['method']==name]
  methods[name]={'samples':len(rows),'median_end_to_end_seconds':statistics.median(r['end_to_end_seconds'] for r in rows),'median_denoise_seconds':statistics.median(r['denoise_seconds'] for r in rows),'max_peak_allocated_GiB':max(r['peak_allocated_GiB'] for r in rows),'NFE':rows[0]['NFE']}
 result['inference_smoke_comparison']={'source':'exact official DMD2 COCO2014 10K package','prompts':4,'seeds_per_prompt':2,'resolution':1024,'methods':methods,'records':values,'quality_claim':'No aggregate quality metric; compare images only as pipeline sanity check'}
assets={}
for path in [*sorted((root/'models/sdxl-base-1.0').rglob('*')),*sorted((root/'datasets/dmd2').glob('*'))]:
 if path.is_file() and path.suffix in ['.json','.txt','.safetensors','.pt','.pkl','.zip']:
  sha=hashlib.sha256()
  with path.open('rb') as f:
   while buf:=f.read(16*1024**2):sha.update(buf)
  assets[str(path.relative_to(root))]={'bytes':path.stat().st_size,'sha256':sha.hexdigest()}
result['asset_sha256_manifest']=assets
(out/'result.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if k in ['status','training_time_seconds','aggregate_examples_per_second','phase_timing','peak_allocated_GiB_per_rank','peak_reserved_GiB_per_rank','nvidia_smi_peak_own_process_MiB_per_gpu']},indent=2))
if 'inference_smoke_comparison' in result:print(json.dumps(result['inference_smoke_comparison']['methods'],indent=2))
