"""One fresh A-PCM10 terminal-prior ablation; training and final student only."""
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
job = root/'outputs/A_pcm10_terminal_low30_g160_20261113'

def run_gpu(cmd, logfile):
    cards = gpu_check([0],76)
    with logfile.open('a') as log:
        proc = subprocess.Popen(cmd,cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        print('STARTED',logfile.name,proc.pid,flush=True)
        try:
            while proc.poll() is None:
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
    save(job/'training_complete.json',dict(G=160,fake=320,checkpoint=str(checkpoint)))
    return checkpoint

def main():
    with (job/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert read(job/'cpu_validation.json')['passed']
        cfg = read(job/'effective_config.json')
        assert cfg['pcm_weight']==10 and cfg['fake_updates']==2 and cfg['batch']==64
        if not (job/'model/status.json').exists():
            if shutil.disk_usage(job).free < 8*1024**3:
                raise RuntimeError('Insufficient room for final student and COCO features; inspect storage')
            if (job/'train.log').exists():
                raise RuntimeError('Inspect previous attempt before restarting')
            run_gpu(command(cfg,root,job.name+'/model',1),job/'train.log')
        checkpoint = verify()
        save(job/'complete.json',dict(G=160,fake=320,checkpoint=str(checkpoint),evaluation='not queued'))
        print('TRAINING_COMPLETE; final student saved',flush=True)

if __name__=='__main__':
    main()
