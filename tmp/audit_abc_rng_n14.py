from pathlib import Path
import collections,hashlib,importlib.util,json,random
root=Path('output/sdxl-crossdistill');out=root/'rng_scope_audit_20261114';e=out/'evidence'
old=root/'a800_BC_paired_quota_20261102/completed'
spec=importlib.util.spec_from_file_location('old_sampler',e/'turning_sampling.py')
mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
result={}
def collision(rows):
 groups=collections.defaultdict(list)
 for item in rows:groups[item['key']].append(item)
 repeated=[v for v in groups.values() if len(v)>1]
 example=next((v for v in repeated if {x['stage'] for x in v}=={2,3}),repeated[0] if repeated else [])
 return dict(draws=len(rows),unique_keys=len(groups),reused_draws=len(rows)-len(groups),
    max_multiplicity=max(map(len,groups.values()),default=0),
    same_update_collision_groups=sum(len({x['update'] for x in v})<len(v) for v in repeated),
    example=example,
    equal_logged_t_pairs=sum(v[i].get('t')==v[0].get('t') for v in repeated for i in range(1,len(v))) if rows and 't' in rows[0] else None)
for arm,folder in [('A','uniform_control'),('B','B'),('C','C')]:
 p=old/folder;cfg=json.loads((p/'config.json').read_text());h=json.loads((p/'source_sha256.json').read_text())
 for name in ['train.py','batching.py','turning_sampling.py']:
  assert hashlib.sha256((e/name).read_bytes()).hexdigest()==h[name],(arm,name)
 rows=[json.loads(x) for x in (p/'rank0.jsonl').read_text().splitlines()]
 gs=[x for x in rows if x['phase']=='student_done'];fs=[x for x in rows if x['phase']=='fake']
 assert len(gs)==160 and len(fs)==320
 seed=cfg['seed'];batch=cfg['batch'];micro=cfg['micro_batch'];ranges=cfg['student_dmd_t_range']
 assert cfg['world_size']==1 and batch==64 and micro==16
 step_stream=[]
 for g in gs:
  idx=g['g']-1
  for k in [g['stage'],4]:
   for pos in range(idx*64,(idx+1)*64):
    step_stream.append(dict(key=seed+200000+1000*k+pos,stage=k,update=idx,pos=pos))
 a=dict(source_hashes_verified=True,student_updates=len(gs),fake_updates=len(fs),
       student_stage_rng=collision(step_stream),
       initial_student_noise=dict(draws=10240,unique_generator_seeds=len({seed+200000+p for p in range(10240)})),
       initial_fake_noise_and_fake_epsilon=dict(draws=20480,unique_generator_seeds=len({seed+100000+p for p in range(20480)})),
       pcm=dict(draws=10240,unique_generator_seeds=10240),prior_enabled=cfg['turning_profile'] is not None)
 if arm!='A':
  profile=e/f'{arm}_profile.json';assert hashlib.sha256(profile.read_bytes()).hexdigest()==h['turning_distribution.json']
  sampler=mod.TurningSampler(profile,979)
  records=collections.defaultdict(list);rates=collections.defaultdict(list);matched=0
  for row in rows:
   ph=row['phase']
   if ph not in ('fake_turning_sample','boundary_turning_sample'):continue
   role='fake' if ph=='fake_turning_sample' else 'student';k=row['stage']
   update=row['fake_update'] if role=='fake' else row['g']
   start=update*64+(0 if role=='fake' else row['micro']*16)
   positions=range(start,start+len(row['timesteps']))
   salt=seed+1000*k+(500000 if role=='fake' else 0)
   expected=sampler.sample(positions,salt,ranges[str(k)])
   assert expected==row['timesteps'],(arm,ph,update,k)
   matched+=len(expected);rates[(role,k)]+=expected
   for pos,t in zip(positions,expected):records[role].append(dict(key=salt+700000+pos,stage=k,update=update,pos=pos,t=t))
  assert matched==40960
  a['prior_log_replay_exact_count']=matched
  a['prior_rng']={role:collision(v) for role,v in records.items()}
  a['prior_rng_role_overlap']=len({x['key'] for x in records['student']} & {x['key'] for x in records['fake']})
  a['observed_rates']={}
  for (role,k),ts in rates.items():
   tt,pp=sampler.distribution(ranges[str(k)])
   a['observed_rates'][f'{role}_{k}']=dict(n=len(ts),high=sum(t>=900 for t in ts)/len(ts),low=sum(t<=100 for t in ts)/len(ts),
     expected_high=sum(p for t,p in zip(tt,pp) if t>=900),expected_low=sum(p for t,p in zip(tt,pp) if t<=100))
  # Noise levels can differ under differing stage supports even with identical uniforms.
  for role,recs in records.items():
   d=collections.defaultdict(list)
   for x in recs:d[x['key']].append(x)
   a['prior_rng'][role]['equal_high_indicator_pairs']=sum((v[i]['t']>=900)==(v[0]['t']>=900) for v in d.values() for i in range(1,len(v)))
 result[arm]=a
(out/'abc_rng_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
for arm,a in result.items():
 print(arm,'student_stage_rng', {k:v for k,v in a['student_stage_rng'].items() if k not in ['example']})
 if a['prior_enabled']:
  print('PRIOR',a['prior_log_replay_exact_count'], {k:{kk:vv for kk,vv in v.items() if kk!='example'} for k,v in a['prior_rng'].items()})
  print('RATES',a['observed_rates'])
