"""Run on the existing H20, only after the refinement arm's complete COCO10K."""
import fcntl
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

from launch import gpu_check

here=Path(__file__).resolve().parent
outputs=Path(os.environ['SDXL_ROOT'])/'outputs'
original=outputs/'h20_C_mix1_terminal_GAN_G160'
direct=outputs/'h20_C_mix1_direct_terminal_GAN_G160'
reference_eval=outputs/(original.name+'_coco10k')

def read(path):
    return json.loads(path.read_text())

def score(folder,method):
    result=read(folder/'complete.json')['scores'][method]
    assert result['n']==10000 and all(math.isfinite(result[k]) for k in ('FID','Patch_FID'))
    return result

def idle():
    while True:
        try:
            gpu_check([0],90)
            return
        except RuntimeError:
            time.sleep(30)

def short_result(probe):
    b=read(probe/'benchmark.json')
    assert (b['status'],b['student_updates'],b['fake_updates'],b['D_updates'],b['effective_batch'],b['accumulation'])==('complete',3,6,3,64,4)
    rows=[json.loads(line) for line in (probe/'rank0.jsonl').read_text().splitlines()]
    gs=[r for r in rows if r['phase']=='student_done']
    fs=[r for r in rows if r['phase'] in ('fake','fake_D')]
    probes=[r for r in rows if r['phase']=='gan_backward_probe']
    assert [r['g'] for r in gs]==[1,2,3] and [r['g'] for r in probes]==[1,2,3] and len(fs)==6
    for i,r in enumerate(gs):
        assert r['stage'] in (2,3) and r['supervised_stages']==[r['stage'],4] and r['GAN_stage']==4
        assert [f['fake_stage'] for f in fs[2*i:2*i+2]]==r['supervised_stages']
        assert all(math.isfinite(r[k]) for k in ('grad_norm','gan_loss','pcm_loss','middle_dm_loss','terminal_dm_loss'))
    assert all(math.isfinite(r[k]) for r in fs for k in ('fake_loss','fake_grad_norm'))
    assert all(math.isfinite(r[k]) for r in fs if r['phase']=='fake_D' for k in ('D_hinge','D_grad_norm','R1_raw'))
    assert all(0<r['weighted_accumulated_latent_rms']<math.inf for r in probes)
    times=b['cycle_seconds']
    assert len(times)==3 and all(0<t<math.inf for t in times)
    return dict(seconds_per_update=sum(times[1:])/2,cycle_seconds=times,peak_allocated_GiB=b['peak_allocated_GiB'],peak_reserved_GiB=b['peak_reserved_GiB'])

def main():
    lock=(here/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    print('WAITING_FOR_ORIGINAL_COCO10K',reference_eval,flush=True)
    while not (reference_eval/'complete.json').is_file():
        if (original/'status.json').exists() and read(original/'status.json').get('status')=='stopped':
            raise RuntimeError('Original training stopped')
        time.sleep(60)
    score(reference_eval,'C_mix1_terminal_GAN')
    assert read(original/'status.json')['student_updates']==160
    idle()
    probe=outputs/'h20_C_mix1_direct_terminal_GAN_probe3'
    if not (probe/'benchmark.json').is_file():
        with (here/'probe_gpu.csv').open('w') as log:
            sample=subprocess.Popen(['nvidia-smi','-i','0','--query-gpu=memory.used,memory.total','--format=csv,noheader,nounits','-l','1'],stdout=log)
            try:
                subprocess.run([sys.executable,'run.py','--short','--execute'],cwd=here,check=True)
            finally:
                sample.terminate()
                sample.wait()
    result=short_result(probe)
    gpu=[list(map(int,line.split(','))) for line in (here/'probe_gpu.csv').read_text().splitlines() if line.strip()]
    result['sampled_device_peak_GiB']=max(row[0] for row in gpu)/1024
    result['sampled_min_free_GiB']=min(row[1]-row[0] for row in gpu)/1024
    print('SHORT_RESULT',result,flush=True)
    (here/'short_result.json').write_text(json.dumps(result,indent=2))
    if result['sampled_min_free_GiB']<2:
        raise RuntimeError('Less than 2 GiB device headroom in short run; review before G160')
    idle()
    if not (direct/'status.json').exists():
        subprocess.run([sys.executable,'run.py','--execute'],cwd=here,check=True)
    state=read(direct/'status.json')
    assert (state['status'],state['student_updates'],state['fake_updates'],state['D_updates'])==('complete',160,320,160)
    print('DIRECT_G160_COMPLETE', direct, flush=True)

if __name__=='__main__':main()
