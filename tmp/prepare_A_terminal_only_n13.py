"""Frozen A source: remove student middle DM; fake and its boundary targets stay."""
from pathlib import Path
import hashlib,json,os,shutil

root=Path(os.environ['SDXL_ROOT']);name='A_pcm1_terminal_dm_g160_20261113'
job=root/'outputs'/name;code=root/'tmp'/name/'code'
base=root/'outputs/turning_BC_paired_quota_20261102/uniform_control'
assert not job.exists(), 'Inspect existing run before preparation'
job.mkdir(parents=True);code.mkdir(parents=True,exist_ok=True)
hashes=json.loads((base/'source_sha256.json').read_text())
for p in (base/'source').glob('*.py'):
    assert hashlib.sha256(p.read_bytes()).hexdigest()==hashes[p.name],p.name
    shutil.copy2(p,code/p.name)
cfg=json.loads((root/'outputs/A_mid_clean_x0/effective_config.json').read_text())
saved=json.loads((base/'config.json').read_text())
for k,v in cfg.items():
    if k=='schedule':assert saved[k]==v+[-1]
    elif k=='data_dir':assert Path(saved[k]).resolve()==(root/v).resolve()
    else:assert saved[k]==v,(k,saved[k],v)
assert cfg['pcm_weight']==1 and cfg['fake_updates']==2 and cfg['pair_fake_stages']
assert cfg['turning_profile'] is None and cfg['gan_weight']==0 and not cfg.get('fake_student_mix',0)

def patch(name,old,new):
    p=code/name;s=p.read_text();assert s.count(old)==1,(name,old)
    p.write_text(s.replace(old,new))

patch('boundary.py', "    if middle not in (2, 3):\n        raise ValueError('Middle phase must be 2 or 3')",
      "    if middle not in (None, 2, 3):\n        raise ValueError('Middle phase must be None, 2 or 3')")
patch('train.py','from turning_sampling import TurningSampler',
      'from turning_sampling import TurningSampler\nfrom pcm_middle_probe_n13 import gradient_diagnostics, PROBE_UPDATES')
patch('train.py','scheduler, condition, rnd["noise"], knots, stage):',
      'scheduler, condition, rnd["noise"], knots, None):')
patch('train.py',"'supervised_stages':[stage,4]", "'supervised_stages':[4]")
patch('train.py','"boundary.py", "sam_pixel.py", "turning_sampling.py"):',
      '"boundary.py", "sam_pixel.py", "turning_sampling.py", "pcm_middle_probe_n13.py"):')
patch('train.py','    if rank == 0:\n        write_json(out / "config.json", config)', '''    config.update(student_stages=[4], dmd_stages=[4], student_dmd_t_range={4:ranges[4]},
                  fake_dmd_t_range=ranges, diagnostic_dm_t_range=ranges,
                  ablation='A: remove student middle DM; retain paired boundary fake',
                  diagnostic_updates=list(PROBE_UPDATES), diagnostic_batch=8)
    config['experiment_modules']['student']='PCM + terminal DM only'
    if rank == 0:
        write_json(out / "config.json", config)''')
# Keep the probe after both fake updates and before normal student accumulation.
patch('train.py','        tick = time.monotonic()\n        stats, originals, boundary_stats = [], [], []', '''        tick = time.monotonic()
        diagnostic_seconds = 0.
        if gcount + 1 in PROBE_UPDATES:
            probe_start=time.monotonic()
            with torch.random.fork_rng(devices=[device.index or 0]):
                ids,cond,real,rnd=training_batch(0,0,'student',4)
                cond=slice_batch(cond,0,8);real=real[:8];rnd=slice_batch(rnd,0,8)
                times={k:slice_batch(phase_random(0,0,k),0,8) for k in (2,3,4)}
                def probe_dm(k):
                    # Middle probes are hypothetical gradients, never optimizer updates.
                    stream=boundary_dm.local_rollout(lambda x,t,c:predict(student,x,t,c),
                        scheduler,cond,rnd['noise'],knots,k if k in (2,3) else None)
                    try:
                        for current,y in stream:
                            if current==k:
                                return boundary_loss(y,k,times[k],cond)
                    finally:
                        stream.close()
                losses={'PCM':lambda:args.pcm_weight*pcm_loss(real,rnd,cond),
                        **{f'DM{k}':lambda k=k:probe_dm(k) for k in (2,3,4)}}
                metrics=gradient_diagnostics(losses,lambda:og.zero_grad(set_to_none=True),
                    lambda:clip_sharded(student,clip=False),device,args.seed)
                torch.cuda.synchronize()
                diagnostic_seconds=time.monotonic()-probe_start
                record(dict(phase='pcm_middle_gradient_probe',g=gcount+1,
                    original_ids=[data['original_indices'][i] for i in ids[:8]],
                    timing='after paired fake updates; before student optimizer',
                    diagnostic_seconds=diagnostic_seconds,**metrics))
                del ids,cond,real,rnd,times,losses,probe_dm,metrics
        stats, originals, boundary_stats = [], [], []''')
patch('train.py','"grad_norm": gnorm, "seconds": time.monotonic() - tick, "cycle_seconds": cycle_seconds,',
      '"grad_norm": gnorm, "gradient_clipped":gnorm > args.grad_clip,\n                "diagnostic_seconds":diagnostic_seconds, "training_cycle_seconds":cycle_seconds-diagnostic_seconds,\n                "seconds": time.monotonic() - tick, "cycle_seconds": cycle_seconds,')
helpers=root/'tmp/A_clean_global_g160_20261113/code'
for n in ['launch.py','turning_batch64_workflow.py']:shutil.copy2(helpers/n,code/n)
shutil.copy2(Path(__file__).with_name('pcm_middle_probe_n13.py'),code/'pcm_middle_probe_n13.py')
changed=sorted(p.name for p in (base/'source').glob('*.py') if p.read_bytes()!=(code/p.name).read_bytes())
assert changed==['boundary.py','train.py']
for p in code.glob('*.py'):compile(p.read_text(),str(p),'exec')
(job/'effective_config.json').write_text(json.dumps(cfg,indent=2))
(job/'baseline_source_hashes.json').write_text(json.dumps(hashes,indent=2))
(job/'design.json').write_text(json.dumps(dict(baseline=str(base),changed_files=changed,
    new_module='pcm_middle_probe_n13.py',student='PCM1+DM4',fake='paired middle boundary+terminal',
    noise_ranges={2:[509,979],3:[259,979],4:[20,979]},GAN=0,mix=0,
    initialization='original SDXL',probe_G=[1,40,80,120,160],probe_batch=8,
    evaluation='fixed training samples only; no new COCO queue'),indent=2))
print('PREPARED',name,changed)
