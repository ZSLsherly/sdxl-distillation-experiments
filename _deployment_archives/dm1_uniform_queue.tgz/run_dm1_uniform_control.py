from pathlib import Path
import fcntl, os, shutil, subprocess, sys, time
import torch
from launch import command, flags, gpu_check
from turning_batch64_workflow import descendants, foreign_on, stop_own
from run_quota_bc import verify
from queue_dm_pair_coco import read, save, score, evaluate_checkpoint, timing
from train import arguments
import boundary as bd
from batching import random_inputs

root=Path(os.environ['SDXL_ROOT'])
runs=root/'outputs'
code=Path(__file__).resolve().parent
base=runs/'c_mix1_first_dm_fake3_g160'
out=runs/'dm1_mid4_uniform_mix1_g160'
prior=runs/'A_pcm10_g160'

def main():
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    original=read(base/'effective_config.json')
    assert original['turning_profile'] and original['match_fake_prior']
    cfg=dict(original,turning_profile=None,match_fake_prior=False)
    sys.argv=['train','--root',str(root),'--run',out.name+'/model',*flags(cfg)]
    a=arguments()
    assert a.fake_updates==3 and a.pair_fake_stages and a.pcm_weight==0 and not a.both_middle
    assert a.fake_mix_start_g==1 and a.fake_student_mix==.01 and a.variant=='dmd_only'
    assert a.max_student_updates==160 and a.batch==64 and a.micro_batch==16
    for g in range(160):
        assert [bd.critic_stage(3*g+i,True,a.seed,3) for i in range(3)]==[1,bd.middle_stage(g,a.seed),4]
    ranges=bd.phase_ranges([*a.schedule,-1],a.boundary_margin,a.boundary_noise_max)
    assert ranges=={1:(759,979),2:(509,979),3:(259,979),4:(20,979)}
    for k,r in ranges.items():
        for phase in ('student','fake'):
            t=torch.cat([random_inputs(range(i,i+64),phase,'cpu',shape=(1,48,48),time_range=r,
                                       seed=a.seed+1000*k)['t'] for i in range(0,2048,64)])
            assert int(t.min())>=r[0] and int(t.max())<=r[1]
            assert abs((t>=900).float().mean().item()-80/(r[1]-r[0]+1))<.045
    save(out/'effective_config.json',cfg)
    print('VERIFIED: uniform stage times, three paired fake updates, source and configuration. WAITING_FOR A_pcm10_g160/comparison.json',flush=True)
    with (prior/'queue.lock').open() as gate:
        fcntl.flock(gate,fcntl.LOCK_SH)
        assert 'A_pcm10' in read(prior/'comparison.json')
    model=out/'model'
    if not (model/'status.json').exists():
        required=sum(p.stat().st_size for p in (base/'model').rglob('*') if p.is_file())+900*1024**2
        if shutil.disk_usage(out).free<required:
            alt=Path('/root')/'dm1_mid4_uniform_mix1_model'
            if shutil.disk_usage(alt.parent).free<required or shutil.disk_usage(out).free<700*1024**2:
                raise RuntimeError('Insufficient disk for model and evaluation')
            alt.mkdir(exist_ok=False)
            model.symlink_to(alt,target_is_directory=True)
        while True:
            if (out/'STOP_REQUEST').exists():return
            try: cards=gpu_check([0],76);break
            except (RuntimeError,subprocess.TimeoutExpired):time.sleep(30)
        with (out/'train.log').open('x') as log:
            proc=subprocess.Popen(command(cfg,root,out.name+'/model',1),cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('STARTED',proc.pid,flush=True)
            try:
                while proc.poll() is None:
                    try: conflicts=foreign_on({cards[0][1]},descendants(proc.pid))
                    except subprocess.TimeoutExpired:
                        time.sleep(10);continue
                    if conflicts:raise RuntimeError('GPU occupied by another process')
                    time.sleep(10)
                if proc.returncode:raise RuntimeError('Training exited '+str(proc.returncode))
            except BaseException:stop_own(proc);raise
    ck=Path(verify(model))
    assert read(model/'data_manifest.json')['train_original_indices']==read(base/'model/data_manifest.json')['train_original_indices']
    assert read(ck/'cpu_validation.json')['strict_load']
    original_eval=runs/'c_mix1_dm_pair_coco10k/DM1_mid4_fake3'
    from queue_dm_pair_coco import reference
    m=read(reference/'manifest.json')
    old=read(original_eval/'manifest.json')
    assert m['samples']==old['samples'] and m['visual_indices']==old['visual_indices']
    result=evaluate_checkpoint('DM1_mid4_uniform',ck,read(model/'config.json'),m,out)
    orig=score(original_eval,'DM1_mid4_fake3')
    save(out/'comparison.json',dict(prior=dict(**orig,**timing(base/'model')),uniform=dict(**result,**timing(model)),delta={k:v-orig[k] for k,v in result.items()}))
    subprocess.run(['bash',str(code/'clip128.sh')],check=True)
    print('COMPLETE',flush=True)
if __name__=='__main__':main()

