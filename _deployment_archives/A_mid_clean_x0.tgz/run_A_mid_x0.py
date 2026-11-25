from pathlib import Path
import os,sys,fcntl,time,shutil
import torch
from diffusers import UNet2DConditionModel, DDIMScheduler
import boundary as bd
from objectives import VPSchedule
from train import arguments
from launch import command,flags
from run_quota_bc import verify
from queue_dm_pair_coco import read,save,reference,score,timing,evaluate_checkpoint
import run_dm1_uniform_control as guard
root=Path(os.environ['SDXL_ROOT'])
runs=root/'outputs'; prior=runs/'A_pcm10_terminal_patchgan'; job=runs/'A_mid_clean_x0'; parent=runs/'turning_BC_paired_quota_20261102/uniform_control'
guard.out=job
def test_targets():
    sched=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
    knots=[999,749,499,249,-1];z=torch.randn(2,4,4,4)
    weights=torch.nn.Parameter(torch.tensor([.01,.02,.03,.04]))
    def predict(x,t,c):return weights[knots.index(int(t[0]))]*x
    for mid in (2,3):
        x=z;expected={}
        with torch.no_grad():
            for k in range(1,5):
                t=torch.full((2,),knots[k-1]);eps=predict(x,t,None)
                expected[k]=sched.clean(x,eps,t)
                x=sched.jump(x,eps,t,knots[k])
                if k==mid: assert not torch.allclose(x,expected[k])
        fake,terminal=bd.rollout_boundary_and_final(predict,sched,None,z,knots,mid)
        torch.testing.assert_close(fake,expected[mid]);torch.testing.assert_close(terminal,x)
        for k,y in bd.local_rollout(predict,sched,None,z,knots,mid):
            torch.testing.assert_close(y,expected[k])
            assert torch.autograd.grad(y.square().mean(),weights)[0].nonzero().flatten().tolist()==[k-1]
with (job/'queue.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    test_targets()
    cfg=read(root/'code/sdxl_research/configs/uniform_control_a800_g160.json')
    sys.argv=['train','--run',job.name+'/model',*flags(cfg)]
    a=arguments()
    assert a.pcm_weight==1 and a.pair_fake_stages and a.turning_profile is None and a.fake_updates==2 and a.gan_weight==0 and a.max_student_updates==160 and a.batch==64
    save(job/'effective_config.json',cfg)
    print('CPU_TARGET_GRADIENT_OK; WAITING_FOR A_pcm10_terminal_patchgan/comparison.json',flush=True)
    with (prior/'queue.lock').open() as gate:
        fcntl.flock(gate,fcntl.LOCK_SH)
        assert 'A_pcm10_patchgan' in read(prior/'comparison.json')
    model=job/'model'
    if not (model/'status.json').exists():
        print('CHECK_DISK: require 7 GiB available on data disk',flush=True)
        while shutil.disk_usage(job).free<7*1024**3:
            if (job/'STOP_REQUEST').exists():sys.exit(0)
            time.sleep(60)
        if (job/'train.log').exists():raise RuntimeError('Inspect existing training log')
        guard.run_gpu(command(cfg,root,job.name+'/model',1),job/'train.log')
    ck=Path(verify(model))
    assert read(parent/'data_manifest.json')['train_original_indices']==read(model/'data_manifest.json')['train_original_indices']
    state=torch.load(ck/'student_rank0.pt',map_location='cpu',mmap=True,weights_only=True)
    with torch.device('meta'): net=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
    net.load_state_dict(state,strict=True,assign=True)
    assert all(torch.isfinite(v).all() for v in state.values())
    del state,net
    save(ck/'cpu_validation.json',dict(strict_load=True,finite=True))
    m=read(reference/'manifest.json')
    a_eval=reference.parent/'coco10k_A'
    assert read(a_eval/'manifest.json')['samples']==m['samples']
    result=evaluate_checkpoint('A_mid_clean_x0',ck,read(model/'config.json'),m,job,run_worker=guard.run_gpu)
    baseline=score(a_eval,next(iter(read(a_eval/'manifest.json')['models'])))
    save(job/'comparison.json',dict(A=baseline,clean=dict(**result,**timing(model)),delta={k:v-baseline[k] for k,v in result.items()}))

