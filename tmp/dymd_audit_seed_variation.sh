set -eu
python3 - <<'PY'
import json,pathlib
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101');p=b/'profiles';pairs=[]
for first,second in [(0,8),(1,9)]:
 if not (p/f'trajectory_{second:02d}.complete').exists():continue
 m0=json.loads((p/f'input_{first:02d}.json').read_text());m1=json.loads((p/f'input_{second:02d}.json').read_text())
 assert m0['condition']==m1['condition'] and m0['seed']==0 and m1['seed']==1
 a=[json.loads(x) for x in (p/f'trajectory_{first:02d}.jsonl').read_text().splitlines()]
 c=[json.loads(x) for x in (p/f'trajectory_{second:02d}.jsonl').read_text().splitlines()]
 assert len(a)==len(c)==360
 assert any(x['norm']!=y['norm'] for x,y in zip(a,c)),(first,second,'identical velocity norms')
 assert any(x.get('kappa')!=y.get('kappa') for x,y in zip(a,c)),(first,second,'identical turning rates')
 pairs.append({'condition':m0['condition'],'samples':[first,second],'seeds':[0,1],'actual_velocity_statistics_differ':True})
result={'pairs_checked':pairs,'interpretation':'Checks that real recorded velocity statistics change across seeds for identical conditioning; not an independent-sample or effect-size claim.'}
(b/'actual_seed_variation_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
PY
