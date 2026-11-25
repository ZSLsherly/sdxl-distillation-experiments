"""One A800 job after the already queued COCO evaluations."""
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from launch import command, flags, gpu_check
from train import arguments
from run_bc_coco_then_winner import env
from turning_batch64_workflow import foreign_on, descendants, stop_own
from queue_dm_pair_coco import evaluate_checkpoint, read, save, reference

root=Path(os.environ['SDXL_ROOT'])
code=Path(__file__).resolve().parent
job=root/'outputs/c_mix1_pcm_dm4_matched_time'
upstream=root/'outputs/c_mix1_dm_pair_coco10k'

def verify(model):
    rows=[json.loads(l) for l in (model/'rank0.jsonl').read_text().splitlines()]
    gs=[r for r in rows if r['phase']=='student_done']
    fs=[r for r in rows if r['phase']=='fake']
    status=read(model/'status.json')
    assert status['status']=='complete' and status['student_updates']==len(gs)>=160
    assert len(fs)==status['fake_updates']==2*len(gs)
    for i,g in enumerate(gs):
        assert g['g']==i+1 and g['supervised_stages']==[4] and not g['gan']
        assert all(math.isfinite(g[k]) for k in ('terminal_dm_loss','pcm_loss','grad_norm'))
        assert g['middle_dm_loss']==0 and g['dmd_loss']==g['terminal_dm_loss']
        assert [r['fake_stage'] for r in fs[2*i:2*i+2]]==[g['stage'],4]
    assert all(math.isfinite(f['fake_loss']) and math.isfinite(f['fake_grad_norm']) for f in fs)
    events=[r for r in rows if r['phase']=='budget_comparison']
    by_g={r['g']:r for r in events}
    assert len(by_g)==len(events) and sum(r['time_milestone'] for r in events)==1
    assert [r['g'] for r in events if r['update_milestone']]==[160]
    checkpoints={read(p)['student_updates']:p.parent for p in (model/'weights').glob('*/complete.json')}
    assert set(checkpoints)==set(by_g)
    return by_g,checkpoints

def main():
    job.mkdir(exist_ok=True)
    lock=(job/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(code/'configs/C_mix1_pcm_terminal_dm.json')
    base=read(root/'outputs/bc_winner_late_mix_20261102/effective_config.json')
    ref=read(upstream/'baseline.json')
    assert cfg==dict(base,terminal_only_dm=True,matched_wall_seconds=cfg['matched_wall_seconds'])
    assert abs(cfg['matched_wall_seconds']-ref['train_wall_hours']*3600)<.01
    sys.argv=['train','--run','check',*flags(cfg)]
    arguments()
    save(job/'effective_config.json',cfg)
    print('VERIFIED_CONFIG; WAITING_FOR',upstream/'complete.json',flush=True)
    # Existing evaluator holds this lock throughout its wait and both evaluations.
    with (upstream/'queue.lock').open() as gate:
        fcntl.flock(gate,fcntl.LOCK_SH)
        done=read(upstream/'complete.json')
        assert set(done)=={'C_mix1','DM234_fake3','DM1_mid4_fake3'}
        for result in done.values():
            assert all(math.isfinite(result[k]) for k in ('FID','Patch_FID'))
    model=job/'model'
    if not (model/'status.json').exists():
        if model.exists(): raise RuntimeError('Incomplete run directory; inspect before retry')
        if shutil.disk_usage(job).free<13*1024**3: raise RuntimeError('Need 13 GiB for two student checkpoints and evaluation')
        while True:
            if (job/'STOP_REQUEST').exists(): return
            try: cards=gpu_check([0],76);break
            except RuntimeError: time.sleep(30)
        with (job/'train.log').open('a') as log:
            p=subprocess.Popen(command(cfg,root,job.name+'/model',1),cwd=code,env=env,
                               stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('TRAIN_STARTED',p.pid,flush=True)
            try:
                while p.poll() is None:
                    if foreign_on({cards[0][1]},descendants(p.pid)): raise RuntimeError('GPU conflict')
                    time.sleep(10)
                if p.returncode: raise RuntimeError(f'Training exited {p.returncode}')
            except BaseException:
                stop_own(p)
                raise
    events,checkpoints=verify(model)
    report={'baseline':ref,'results':{}}
    for g,ck in sorted(checkpoints.items()):
        ev=events[g]
        if ev['time_milestone']: assert ev['train_seconds']>=cfg['matched_wall_seconds']
        scores=evaluate_checkpoint(f'PCM_DM4_G{g}',ck,read(model/'config.json'),read(reference/'manifest.json'),job)
        report['results'][g]=dict(**scores,**ev,delta_FID=scores['FID']-ref['FID'],
                                   delta_Patch_FID=scores['Patch_FID']-ref['Patch_FID'])
        save(job/'comparison.json',report)
    save(job/'complete.json',report)
    print('COMPLETE',flush=True)
if __name__=='__main__': main()

