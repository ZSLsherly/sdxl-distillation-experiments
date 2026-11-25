"""Run one Flash-style terminal GAN study serially after the selected 1% control."""
import fcntl,json,math,os,shutil,subprocess,sys,time
from pathlib import Path
from launch import command,gpu_check
from turning_batch64_workflow import descendants,foreign_on,stop_own

root=Path(os.environ['SDXL_ROOT']); work=Path(os.environ['SDXL_WORK'])
prior=root/'outputs/bc_winner_late_mix_20261102'; job=root/'outputs/flash_gan_g160_20261102'
code=Path(__file__).resolve().parent
def load(p): return json.loads(p.read_text())
def write(p,v):
    tmp=p.with_suffix('.new');tmp.write_text(json.dumps(v,indent=2));tmp.replace(p)
def check_stop():
    if any((d/n).exists() for d in (job,prior) for n in ('STOP_REQUEST','USER_STOP')):
        raise RuntimeError('Stop requested')
def make_config(winner,recipe,micro):
    cfg=load(code/'configs'/f'turning_{winner}_quota_a800_g160.json')
    cfg.update(variant='joint4',discriminator='sam',sam_path=str(work/'flash_gan_inputs/sam-vit-base'),
               gan_view='whole',gan_augment='none',gan_loss='hinge',flash_terminal_gan=True,
               both_middle=recipe['both_middle'],gan_weight=.005,d_lr=.0001,gan_ramp_updates=0,
               gradient_probe_every=40,micro_batch=micro,fake_student_mix=recipe['student_mix'],
               fake_track_pilot=bool(recipe['student_mix']),fake_mix_start_g=1)
    return cfg
def launch_trial(cfg,path,lf):
    while True:
        check_stop()
        try: cards=gpu_check([0],76);break
        except RuntimeError:time.sleep(30)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
    with lf.open('x') as log:
        p=subprocess.Popen(command(cfg,root,f'{job.name}/{path.name}',1),cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('TRAIN_STARTED',path.name,p.pid,flush=True)
        try:
            while p.poll() is None:
                check_stop()
                if foreign_on({cards[0][1]},descendants(p.pid)):raise RuntimeError('GPU shared by another process')
                time.sleep(10)
            return p.returncode
        except BaseException:stop_own(p);raise
def verify(path,cfg):
    s=load(path/'status.json')
    assert (s['status'],s['student_updates'],s['fake_updates'],s['D_updates'],s['effective_batch'])==('complete',160,320,160,64)
    rows=[json.loads(l) for l in (path/'rank0.jsonl').read_text().splitlines()]
    gs=[r for r in rows if r['phase']=='student_done'];fs=[r for r in rows if r['phase'] in ('fake','fake_D')]
    assert len(gs)==160 and len(fs)==320
    for i,g in enumerate(gs):
        mid=[2,3] if cfg['both_middle'] else [g['stage']]
        assert g['g']==i+1 and g['supervised_stages']==mid+[4] and g['GAN_stage']==4
        if cfg['both_middle']: assert [v['fake_supervised_stages'] for v in fs[2*i:2*i+2]]==[mid,[4]]
        else: assert [v['fake_stage'] for v in fs[2*i:2*i+2]]==mid+[4]
        assert all(math.isfinite(g[k]) for k in ('grad_norm','pcm_loss','middle_dm_loss','terminal_dm_loss','gan_loss'))
        for f in fs[2*i:2*i+2]: assert all(math.isfinite(f[k]) for k in ('fake_loss','fake_grad_norm'))
    probes=[v for v in rows if v['phase']=='boundary_GAN_gradient_probe']
    assert probes and all(v['gan_input']=='detached_terminal_refinement' and v['GAN_latent_rms']>0 and math.isfinite(v['GAN_latent_rms']) for v in probes)
    w=list((path/'weights').glob('*/complete.json'));assert len(w)==1 and load(w[0])['student_updates']==160
    ev=list((path/'eval').glob('*_g000160_*/rank0.json'));assert ev and len(load(ev[-1])['samples'])==8
    assert all((ev[-1].parent/i['image']).is_file() for i in load(ev[-1])['samples'])
    import torch
    from diffusers import UNet2DConditionModel
    f=w[0].parent/'student_rank0.pt'
    state=torch.load(f,map_location='cpu',mmap=True,weights_only=True)
    with torch.device('meta'): model=UNet2DConditionModel.from_config(load(root/'models/sdxl-base-1.0/unet/config.json'))
    model.load_state_dict(state,strict=True,assign=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    return str(f)
def main():
    job.mkdir(exist_ok=True)
    with (job/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (job/'complete.json').exists():return
        print('WAITING_FOR_WINNER_MIX_G160',flush=True)
        while not (prior/'training_complete.json').is_file():check_stop();time.sleep(30)
        check_stop()
        for arm,folder in [('B','coco10k'),('C','coco10k'),('A','coco10k_A')]:
            score=load(prior/folder/'complete.json')['scores'][arm]
            assert score['n']==10000 and all(math.isfinite(score[k]) for k in ('FID','Patch_FID'))
        winner=load(prior/'selection.json')['winner'];assert winner in ('B','C') and load(prior/'training_complete.json')['winner']==winner
        recipe=load(job/'recipe.json')
        assert recipe['student_mix'] in (0,.01) and isinstance(recipe['both_middle'],bool)
        if shutil.disk_usage(job).free<8*1024**3:raise RuntimeError('Insufficient disk for final student')
        for micro in (4,2,1):
            path=job/f'gan_micro{micro}'
            cfg=make_config(winner,recipe,micro)
            lf=job/(path.name+'.log')
            if not path.exists() and not lf.exists():
                write(job/'effective_config.json',cfg)
                rc=launch_trial(cfg,path,lf)
                print('TRAIN_EXIT',path.name,rc,flush=True)
            if (path/'status.json').is_file() and load(path/'status.json')['status']=='complete':
                weight=verify(path,cfg)
                write(job/'complete.json',dict(winner=winner,config=cfg,weight=weight,status=load(path/'status.json')))
                print('GAN_G160_VERIFIED',weight,flush=True);return
            logs=lf.read_text(errors='replace') if lf.exists() else ''
            rows=path/'rank0.jsonl'
            begun=rows.exists() and any(json.loads(l)['phase']=='student_done' for l in rows.read_text().splitlines() if l.strip())
            if begun or 'out of memory' not in logs.lower():raise RuntimeError(f'Inspect {lf}; automatic retry is limited to initial CUDA OOM')
        raise RuntimeError('Initial CUDA OOM even at microbatch 1')
if __name__=='__main__':main()
