set -eu
/data1/shenglin.zeng/dymd_prior_8_20261101/env/bin/python - <<'PY'
import json,math,pathlib,collections
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
plan=json.loads((b/'sample_plan.json').read_text());audited=[]
for marker in sorted((b/'profiles').glob('trajectory_*.complete')):
 idx=int(marker.stem.split('_')[-1]);meta=json.loads((b/'profiles'/f'input_{idx:02d}.json').read_text())
 assert meta['sample']==idx and meta['seed']==plan[idx]['seed'] and meta['condition']==plan[idx]['condition']
 rows=[json.loads(line) for line in marker.with_suffix('.jsonl').read_text().splitlines()]
 assert len(rows)==360,(idx,len(rows))
 groups=collections.defaultdict(list)
 for r in rows:
  assert r['sample']==idx and r['shape'][1]==21 and r['norm']>0 and math.isfinite(r['norm'])
  groups[r['branch'],r['centering']].append(r)
 assert len(groups)==9
 for key,g in groups.items():
  assert [r['step'] for r in g]==list(range(40)),(idx,key)
  assert sum('kappa' in r for r in g)==39
  for i,r in enumerate(g[1:],1):
   assert math.isfinite(r['kappa']) and r['kappa']>=0
   assert g[i-1]['sigma']>r['sigma']
   assert abs(r['kappa']*r['delta_sigma']-r['angle_rad'])<1e-9
 audited.append({'sample':idx,'condition':meta['condition'],'seed':meta['seed'],'rows':len(rows)})
report={'target':128,'completed_validated':len(audited),'samples':audited,'video_count':len(list((b/'videos').rglob('*.mp4'))),'note':'Official engine decodes videos in groups of world_size; complete trajectory markers may precede video export.'}
(b/'completed_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
PY
