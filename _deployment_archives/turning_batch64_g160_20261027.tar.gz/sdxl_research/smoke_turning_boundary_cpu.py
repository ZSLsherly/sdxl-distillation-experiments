"""Synthetic tiny-model autograd check for both priors; not SDXL/GPU training."""
import copy,json,time
from pathlib import Path
import torch
from diffusers import UNet2DConditionModel,DDIMScheduler
from objectives import VPSchedule
from turning_sampling import TurningSampler
import boundary as bd

def run(profile):
    torch.manual_seed(71);torch.set_num_threads(2)
    student=UNet2DConditionModel(sample_size=16,in_channels=4,out_channels=4,
        down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),
        block_out_channels=(16,32),layers_per_block=1,norm_num_groups=4,cross_attention_dim=16,attention_head_dim=4,
        addition_embed_type='text_time',addition_time_embed_dim=4,projection_class_embeddings_input_dim=32)
    fake=copy.deepcopy(student);teacher=copy.deepcopy(student).eval().requires_grad_(False)
    initial=[p.detach().clone() for p in student.parameters()]
    schedule=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
    knots=[999,749,499,249,-1];ranges=bd.phase_ranges(knots);sampler=TurningSampler(profile)
    og=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(0,.999),weight_decay=0)
    of=torch.optim.AdamW(fake.parameters(),lr=5e-6,betas=(0,.999),weight_decay=0)
    def predict(model,x,t,c,uncond=False):
        pe,pp,ti=c
        return model(x,t,encoder_hidden_states=torch.zeros_like(pe) if uncond else pe,
            added_cond_kwargs={'text_embeds':torch.zeros_like(pp) if uncond else pp,'time_ids':ti},return_dict=False)[0]
    def guided(x,t,c):
        un=predict(teacher,x,t,c,True);return un+8*(predict(teacher,x,t,c)-un)
    def batch():return torch.randn(2,4,16,16),(torch.randn(2,3,16),torch.randn(2,8),torch.ones(2,6)*64)
    fcount=0;rows=[];start=time.monotonic()
    for g in range(4):
        for _ in range(2):
            n,c=batch();k=bd.critic_stage(fcount)
            y,_=bd.rollout_boundary_and_final(lambda x,t,c:predict(student,x,t,c),schedule,c,n,knots,k)
            lo,hi=ranges[k];t=torch.randint(lo,hi+1,(2,));eps=torch.randn_like(y)
            noised=bd.renoise(schedule,y,knots[k],t,eps)
            of.zero_grad(set_to_none=True)
            loss=bd.fake_loss(schedule,y,knots[k],t,eps,predict(fake,noised,t,c))
            loss.backward();torch.nn.utils.clip_grad_norm_(fake.parameters(),1,error_if_nonfinite=True)
            of.step();of.zero_grad(set_to_none=True);fcount+=1
        n,c=batch();middle=bd.middle_stage(g,20261017);og.zero_grad(set_to_none=True);times={}
        for k,y in bd.local_rollout(lambda x,t,c:predict(student,x,t,c),schedule,c,n,knots,middle):
            lo,hi=ranges[k];t=torch.randint(lo,hi+1,(2,));eps=torch.randn_like(y)
            # Consume baseline RNG first, then only replace t on a separate stream.
            t=t.new_tensor(sampler.sample([g*2,g*2+1],20261017+1000*k,ranges[k]));times[k]=t.tolist()
            assert lo<=int(t.min())<=int(t.max())<=hi
            with torch.no_grad():
                xt=bd.renoise(schedule,y.detach(),knots[k],t,eps)
                ef=predict(fake,xt,t,c);et=guided(xt,t,c)
            dm,direction=bd.dm_surrogate(schedule,y,knots[k],t,xt,ef,et)
            assert not direction.requires_grad;dm.backward()
        real=torch.randn_like(n);t=torch.randint(770,1000,(2,));eps=torch.randn_like(real)
        xh=schedule.noise(real,eps,t)
        with torch.no_grad():
            xk=xh
            for j in range(2):
                tj=t-j*10;xk=schedule.jump(xk,guided(xk,tj,c),tj,tj-10)
            target=schedule.jump(xk,predict(student,xk,t-20,c),t-20,749)
        pcm=(schedule.jump(xh,predict(student,xh,t,c),t,749)-target).square().mean();pcm.backward()
        norm=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True)
        og.step();og.zero_grad(set_to_none=True)
        assert all(p.grad is None for model in [teacher,fake] for p in model.parameters())
        rows.append({'G':g+1,'fake_updates':fcount,'middle':middle,'times':times,'pcm':float(pcm),'student_grad_norm':float(norm)})
    assert any(not torch.equal(a,b) for a,b in zip(initial,student.parameters()))
    return {'passed':True,'G_updates':4,'fake_updates':fcount,'GAN_loaded':False,'wall_seconds':time.monotonic()-start,'rows':rows}

if __name__=='__main__':
    folder=Path(__file__).parent
    result={'scope':'CPU random tiny UNets and synthetic data; real autograd, not pretrained SDXL or FSDP/CUDA validation',
        'arms':{n:run(folder/'turning_profiles'/f'{n}_512.json') for n in ['normalized_u','raw_angle']}}
    out=folder.parent/'output/sdxl-crossdistill/turning_ablation_20261027/boundary_cpu_smoke.json'
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{a:b for a,b in v.items() if a!='rows'} for k,v in result['arms'].items()},indent=2))
