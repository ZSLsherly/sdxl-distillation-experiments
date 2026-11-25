"""A800: archived A trainer and single-variable PCM-weight ablation."""
from pathlib import Path
import fcntl, hashlib, json, math, os, shutil, subprocess, sys, time
import torch
from diffusers import UNet2DConditionModel
from launch import command, flags, gpu_check
from run_quota_bc import verify
from turning_batch64_workflow import descendants, foreign_on, stop_own
from queue_dm_pair_coco import evaluate_checkpoint, read, reference, save, score, timing
from train import arguments

root=Path(os.environ['SDXL_ROOT'])
base=root/'outputs/turning_BC_paired_quota_20261102/uniform_control'
out=root/'outputs/A_pcm10_g160'
prior=root/'outputs/c_mix1_pcm_dm4_matched_time'
code=Path(__file__).resolve().parent

def main():
    lock=(out/'queue.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=read(root/'code/sdxl_research/configs/uniform_control_a800_g160.json')
    old=read(base/'config.json')
    hashes=read(base/'source_sha256.json')
    for name,digest in hashes.items():
        assert hashlib.sha256((code/name).read_bytes()).hexdigest()==digest,name
    for k,v in cfg.items():
        expected=(root/v).as_posix() if k=='data_dir' else v+[-1] if k=='schedule' else v
        assert old[k]==expected,(k,v,old[k])
    assert cfg['pcm_weight']==1 and cfg['turning_profile'] is None and cfg['gan_weight']==0
    cfg=dict(cfg,pcm_weight=10)
    sys.argv=['train','--root',str(root),'--run',out.name+'/model',*flags(cfg)]
    a=arguments()
    assert a.pair_fake_stages and a.fake_updates==2 and a.max_student_updates==160 and a.batch==64
    save(out/'effective_config.json',cfg)
    a_eval=reference.parent/'coco10k_A'
    a_manifest=read(a_eval/'manifest.json')
    m=read(reference/'manifest.json')
    assert a_manifest['samples']==m['samples'] and a_manifest['visual_indices']==m['visual_indices'] and a_manifest['protocol']==m['protocol']
    baseline=score(a_eval,next(iter(a_manifest['models'])))
    print('VERIFIED: original A source hashes and launch config; only pcm_weight 1 -> 10; WAITING_FOR_G160_COCO',flush=True)
    with (prior/'queue.lock').open() as gate:
        fcntl.flock(gate,fcntl.LOCK_SH)
        previous_score=read(prior/'g160_scores.json')
        assert all(math.isfinite(previous_score[k]) for k in ('FID','Patch_FID'))
    model=out/'model'
    if not model.exists():
        if shutil.disk_usage(out).free<8*1024**3: raise RuntimeError('Insufficient space for weights and evaluation')
        while True:
            if (out/'STOP_REQUEST').exists():return
            try: cards=gpu_check([0],76);break
            except (RuntimeError,subprocess.TimeoutExpired):time.sleep(30)
        with (out/'train.log').open('x') as log:
            proc=subprocess.Popen(command(cfg,root,out.name+'/model',1),cwd=code,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            print('TRAIN_STARTED',proc.pid,flush=True)
            try:
                while proc.poll() is None:
                    try: conflict=foreign_on({cards[0][1]},descendants(proc.pid))
                    except subprocess.TimeoutExpired:
                        print('GPU_QUERY_TIMEOUT; retrying',flush=True)
                        time.sleep(10)
                        continue
                    if conflict:raise RuntimeError('Foreign GPU process')
                    time.sleep(10)
                if proc.returncode: raise RuntimeError(f'Training exit {proc.returncode}')
            except BaseException: stop_own(proc);raise
    ck=Path(verify(model))
    assert read(model/'data_manifest.json')['train_original_indices']==read(base/'data_manifest.json')['train_original_indices']
    state=torch.load(ck/'student_rank0.pt',map_location='cpu',mmap=True,weights_only=True)
    with torch.device('meta'):
        net=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
    net.load_state_dict(state,strict=True,assign=True)
    assert all(torch.isfinite(t).all() for t in state.values())
    del net,state
    save(ck/'cpu_validation.json',dict(strict_load=True,finite=True))
    result=evaluate_checkpoint('A_pcm10',ck,read(model/'config.json'),m,out)
    save(out/'comparison.json',dict(A=dict(**baseline,**timing(base)),A_pcm10=dict(**result,**timing(model)),
         delta={k:result[k]-baseline[k] for k in baseline}))
    print('COMPLETE',flush=True)
if __name__=='__main__': main()

