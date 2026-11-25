"""Clean-global high30 prior ablation; identical trainer and one fresh G160 run."""
from pathlib import Path
import fcntl
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from launch import command, gpu_check
from turning_batch64_workflow import descendants, foreign_on, stop_own
def read(path):
    return json.loads(path.read_text())

def save(path, value):
    path.write_text(json.dumps(value, indent=2))

root = Path(os.environ['SDXL_ROOT'])
code = Path(__file__).resolve().parent
job = root/'outputs/A_clean_global_high30_g160_20261114_v2'

def run_gpu(cmd, logfile):
    cards = gpu_check([0],76)
    with logfile.open('a') as log:
        proc = subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('STARTED',logfile.name,proc.pid,flush=True)
        try:
            checked_first = False
            while proc.poll() is None:
                if not checked_first and (job/'model/status.json').exists():
                    state=read(job/'model/status.json')
                    if state.get('student_updates',0)>=1:
                        first_update_check()
                        checked_first=True
                try:
                    conflicts = foreign_on({cards[0][1]},descendants(proc.pid))
                except subprocess.TimeoutExpired:
                    time.sleep(10)
                    continue
                if conflicts:
                    raise RuntimeError('Another GPU process appeared')
                if (job/'STOP_REQUEST').exists() and logfile.name != 'train.log':
                    raise RuntimeError('Evaluation stop requested')
                time.sleep(10)
            if proc.returncode:
                raise RuntimeError(f'{logfile.name} exited {proc.returncode}')
        except BaseException:
            stop_own(proc)
            raise

def verify():
    model = job/'model'
    state = read(model/'status.json')
    assert (state['status'],state['student_updates'],state['fake_updates']) == ('complete',160,320)
    rows = [json.loads(l) for l in (model/'rank0.jsonl').read_text().splitlines()]
    gs = [r for r in rows if r['phase']=='student_done']
    fs = [r for r in rows if r['phase']=='fake']
    assert [g['g'] for g in gs] == list(range(1,161)) and len(fs)==320
    for i,g in enumerate(gs):
        assert g['supervised_stages']==[g['stage'],4] and not g['gan']
        assert [f['fake_stage'] for f in fs[2*i:2*i+2]]==[g['stage'],4]
        assert all(math.isfinite(g[k]) for k in ('pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm'))
    assert all(math.isfinite(f['fake_loss']) and math.isfinite(f['fake_grad_norm']) for f in fs)
    save(job/'sampling_validation.json',check_samples(rows))
    checkpoints = list((model/'weights').glob('*/complete.json'))
    assert len(checkpoints)==1 and read(checkpoints[0])['student_updates']==160
    checkpoint = checkpoints[0].parent
    import torch
    from diffusers import UNet2DConditionModel
    weights = torch.load(checkpoint/'student_rank0.pt',map_location='cpu',weights_only=True,mmap=True)
    with torch.device('meta'):
        net = UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
    net.load_state_dict(weights,strict=True,assign=True)
    assert all(torch.isfinite(v).all() for v in weights.values())
    save(checkpoint/'cpu_validation.json',dict(strict_load=True,finite=True))
    evals=sorted((model/'eval').glob('*_g000160_*'))
    assert evals
    meta=read(evals[-1]/'rank0.json')
    assert len(meta['samples'])==8 and all((evals[-1]/x['image']).exists() for x in meta['samples'])
    save(job/'training_complete.json',dict(G=160,fake=320,checkpoint=str(checkpoint)))
    return checkpoint

def check_samples(rows,first_only=False):
    groups={}
    for row in rows:
        phase=row['phase']
        if phase not in ('boundary_turning_sample','fake_turning_sample'):
            continue
        if first_only and ((phase=='boundary_turning_sample' and row['g']!=0) or
                           (phase=='fake_turning_sample' and row['fake_update']>=2)):
            continue
        name=('student' if phase=='boundary_turning_sample' else 'fake')+'_'+str(row['stage'])
        ts=row['timesteps']
        assert all(isinstance(t,int) and 20<=t<=979 for t in ts)
        groups.setdefault(name,[]).extend(ts)
    assert groups
    if not first_only:
        assert set(groups)=={f'{p}_{k}' for p in ('student','fake') for k in (2,3,4)}
        for p in ('student','fake'):
            assert len(groups[p+'_2'])+len(groups[p+'_3'])==160*64
            assert len(groups[p+'_4'])==160*64
    return {k:dict(n=len(ts),high_fraction=sum(t>=900 for t in ts)/len(ts),minimum=min(ts),maximum=max(ts)) for k,ts in groups.items()}


def first_update_check():
    model=job/'model'
    cfg=read(model/'config.json')
    assert cfg['student_dmd_t_range']=={str(k):[20,979] for k in (2,3,4)}
    assert cfg['pcm_weight']==1 and cfg['effective_batch']==64 and cfg['accumulation']==4
    base=root/'outputs/A_clean_global_g160_20261113/model'
    assert read(model/'data_manifest.json')['train_original_indices']==read(base/'data_manifest.json')['train_original_indices']
    text=(model/'rank0.jsonl').read_text()
    rows=[json.loads(x) for x in text[:text.rfind('\n')].splitlines()]
    g=next(r for r in rows if r['phase']=='student_done')
    fs=[r for r in rows if r['phase']=='fake'][:2]
    assert [f['fake_stage'] for f in fs]==[g['stage'],4]
    for row,keys in [(g,('pcm_loss','middle_dm_loss','terminal_dm_loss','grad_norm'))]+[(f,('fake_loss','fake_grad_norm')) for f in fs]:
        assert all(math.isfinite(row[k]) for k in keys)
    inputs=[r for r in rows if r['phase']=='fake_boundary_input'][:2]
    assert len(inputs)==2 and all(r['boundary_t']==-1 and r['renoise_range']==[20,979] for r in inputs)
    assert cfg['match_fake_prior'] and cfg['turning_sampling']['prior_intervals']==[[900,979]]
    assert cfg['turning_sampling']['interval_rule']=='quota_uniform'
    sample_stats=check_samples(rows,first_only=True)
    save(job/'first_update_validation.json',dict(passed=True,G=g['g'],fake_stages=[f['fake_stage'] for f in fs],same_train_data=True,noise_sampling=sample_stats))
    print('FIRST_BATCH64_UPDATE_VALIDATED',flush=True)


def main():
    with (job/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(job/'cpu_validation.json')['passed']
        prior=root/'outputs/A_pcm1_terminal_dm_g160_20261113'
        print('WAITING_FOR',prior.name,flush=True)
        with (prior/'queue.lock').open('a') as gate:
            fcntl.flock(gate,fcntl.LOCK_SH)
            done=read(prior/'complete.json')
            assert done['G']==160 and done['fake']==320
        if (job/'STOP_REQUEST').exists():
            print('CANCELLED_BEFORE_TRAINING',flush=True)
            return
        cfg = read(job/'effective_config.json')
        base_cfg=read(root/'outputs/A_clean_global_g160_20261113/effective_config.json')
        assert {k for k in cfg if cfg[k]!=base_cfg[k]}=={'turning_profile','match_fake_prior'}
        assert cfg['match_fake_prior']
        assert cfg['pcm_weight']==1 and cfg['fake_updates']==2 and cfg['batch']==64
        if not (job/'model/status.json').exists():
            if shutil.disk_usage(job).free < 8*1024**3:
                raise RuntimeError('Insufficient room for final student and fixed training previews; inspect storage')
            if (job/'train.log').exists():
                raise RuntimeError('Inspect previous attempt before restarting')
            run_gpu(command(cfg,root,job.name+'/model',1),job/'train.log')
        checkpoint = verify()
        save(job/'complete.json',dict(G=160,fake=320,checkpoint=str(checkpoint),evaluation='not queued'))
        print('TRAINING_COMPLETE; final student saved',flush=True)

if __name__=='__main__':
    main()
