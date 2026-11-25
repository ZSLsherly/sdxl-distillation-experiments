import json, pathlib, re, subprocess, shutil, hashlib, time
root=pathlib.Path(__file__).resolve().parent
pf=root/'official/pf_wan'
sha=subprocess.check_output(['git','-C',str(pf),'rev-parse','HEAD'],text=True).strip()
assert sha=='6b407f5776479d17fa3574c3697e42e738b2ba0c',sha
cfg=json.loads(re.sub(r'//[^\n]*','',(pf/'configs/generate/pf_wan_i2v.jsonc').read_text(encoding='utf-8')))
lines=(pf/'assets/examples/manifest.txt').read_text(encoding='utf-8').splitlines()[:8]
assert len(lines)==8
for line in lines:
 p=pf/line.split('@@')[1].strip()
 assert p.exists(),p
 assert not p.read_bytes().startswith(b'version https://git-lfs'),f'LFS missing: {p}'
plan=[]; manifest=[];completed=[]
inputs=root/'inputs128';inputs.mkdir(exist_ok=True)
for seed in range(16):
 for condition,line in enumerate(lines):
  prompt,image=line.split('@@');source=pf/image.strip();idx=seed*8+condition
  dest=inputs/f'sample_{idx:03d}_condition_{condition:02d}_seed_{seed:02d}{source.suffix}'
  shutil.copyfile(source,dest)
  plan.append({'sample':idx,'condition':condition,'seed':seed,'prompt':prompt,'filename':str(dest),'source_image':image.strip(),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()})
  marker=root/'profiles'/f'trajectory_{idx:02d}.complete'
  if marker.exists():
   saved=[json.loads(x) for x in marker.with_suffix('.jsonl').read_text().splitlines()]
   assert len(saved)==360 and {r['sample'] for r in saved}=={idx},idx
   meta=json.loads((root/'profiles'/f'input_{idx:02d}.json').read_text())
   assert meta['seed']==seed and meta['condition']==condition,meta
   completed.append(idx)
  else:
   interrupted=[root/'profiles'/f'trajectory_{idx:02d}.jsonl',root/'profiles'/f'input_{idx:02d}.json']
   existing=[p for p in interrupted if p.exists()]
   if existing:
    archive=root/'interrupted_attempts'/f'{time.time_ns()}_{idx:03d}';archive.mkdir(parents=True)
    for p in existing:p.rename(archive/p.name)
   manifest.append(prompt+'@@'+str(dest))
(root/'sample_plan.json').write_text(json.dumps(plan,indent=2))
(pf/'assets/examples/prior128.txt').write_text('\n'.join(manifest)+'\n')
cfg['inference']['positive_prompt']='./assets/examples/prior128.txt'
cfg['meta_model']['hybrid_gpu_num']=4
cfg['meta_model']['ulysses_size']=4
cfg['output_dir']=str(root/'videos')
cfg['proj_name']='official_prior128'
(pf/'configs/generate/prior128.jsonc').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
(root/'protocol.json').write_text(json.dumps({'official_code_sha':sha,'sample_selection':'first 8 official examples, fixed before any outputs','seed':0,'steps':40,'shift':5,'cfg':5,'frames':81,'max_area':307200,'gpu_count':4,'model_revision':'8fa7e17d5f944d0275df7dc7929137b956367ff8','measure':'acos(dot(unit(center_future(v_k)),unit(center_future(v_k+1)))) / abs(delta_sigma)','branches':['cfg5','conditional','negative'],'deviations':['8 trajectories vs paper 32','official example inputs, not undisclosed paper profiling inputs','4 GPUs vs official default 8','custom read-only measurement hook; DyMD profiling code not located','smoothing and epsilon not disclosed; raw curves are primary']},ensure_ascii=False,indent=2))
protocol=json.loads((root/'protocol.json').read_text())
protocol.update(target_trajectories=128,unique_conditions=8,seeds=list(range(16)),sample_selection='first 8 official examples x seeds 0..15; seed-major order fixed before outcomes',seed='0..15',uncertainty_unit='condition; 16 seeds per condition are correlated, not 128 distinct scenes')
protocol['deviations'][0]='128 trajectories on 8 conditions x 16 seeds, versus paper 32 undisclosed trajectories'
(root/'protocol.json').write_text(json.dumps(protocol,indent=2))
print('PREPARED',sha,len(plan),'trajectories',len(lines),'conditions',len(completed),'already completed')
