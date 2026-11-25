"""Four tiny real-autograd updates, using actual diffusers/SAM architectures.

Random small models and synthetic data ONLY. No GPU, downloads, quality or
throughput claims. Exercises the same objective/rollout/decoder/D functions.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import torch
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
from transformers import SamVisionConfig
from transformers.models.sam.modeling_sam import SamVisionEncoder
from objectives import VPSchedule
from image_views import decode_view
from sam_pixel import SAMDiscriminator
from gan_losses import discriminator_loss, generator_loss
import boundary as bd


def run(flash=False, paired_mix=False):
    torch.set_num_threads(2)
    torch.manual_seed(71)
    student=UNet2DConditionModel(sample_size=16,in_channels=4,out_channels=4,
        down_block_types=('CrossAttnDownBlock2D','DownBlock2D'),
        up_block_types=('UpBlock2D','CrossAttnUpBlock2D'),block_out_channels=(16,32),
        layers_per_block=1,norm_num_groups=4,cross_attention_dim=16,attention_head_dim=4,
        addition_embed_type='text_time',addition_time_embed_dim=4,
        projection_class_embeddings_input_dim=32)
    fake=copy.deepcopy(student)
    if paired_mix:
        bd.add_fake_stage_condition(fake)
    teacher=copy.deepcopy(student).eval().requires_grad_(False)
    vae=AutoencoderKL(block_out_channels=(16,16,32),latent_channels=4,
        down_block_types=('DownEncoderBlock2D',)*3,up_block_types=('UpDecoderBlock2D',)*3,
        norm_num_groups=4,layers_per_block=1).eval().requires_grad_(False)
    vae.enable_gradient_checkpointing()
    v=SamVisionConfig(hidden_size=32,num_hidden_layers=12,num_attention_heads=4,
        output_channels=16,image_size=64,patch_size=8,window_size=4,
        global_attn_indexes=[2,5,8,11],mlp_dim=64)
    disc=SAMDiscriminator(backbone=SamVisionEncoder(v))
    schedule=VPSchedule(DDIMScheduler(beta_start=.00085,beta_end=.012,beta_schedule='scaled_linear').alphas_cumprod)
    knots=[999,749,499,249,-1];ranges=bd.phase_ranges(knots)
    og=torch.optim.AdamW(student.parameters(),lr=1e-6,betas=(0,.999),weight_decay=0)
    of=torch.optim.AdamW(fake.parameters(),lr=5e-6,betas=(0,.999),weight_decay=0)
    od=torch.optim.AdamW(disc.trainable(),lr=1e-4,betas=(0,.99),weight_decay=0)
    def predict(model,x,t,c,stage=None):
        return model(x,t,encoder_hidden_states=c['prompt'],
            class_labels=bd.stage_labels(stage,x) if model is fake and paired_mix else None,
            added_cond_kwargs={'text_embeds':c['pooled'],'time_ids':c['time_ids']},return_dict=False)[0]
    def batch():
        return torch.randn(2,4,16,16),dict(prompt=torch.randn(2,3,16),pooled=torch.randn(2,8),time_ids=torch.ones(2,6)*64)
    def dm(y,k,c):
        u=torch.randint(ranges[k][0],ranges[k][1]+1,(len(y),))
        with torch.no_grad():
            x=bd.renoise(schedule,y.detach(),knots[k],u,torch.randn_like(y))
            ef=predict(fake,x,u,c,k);et=predict(teacher,x,u,c)
        return bd.dm_surrogate(schedule,y,knots[k],u,x,ef,et)[0]
    rows=[];fcount=dcount=0
    initial_student=[p.detach().clone() for p in student.parameters()]
    begin=time.monotonic()
    for g in range(4):
        for repeat in range(2):
            noise,c=batch();k=bd.critic_stage(fcount,paired=flash)
            both=flash and not paired_mix and repeat==0
            views=bd.rollout_boundary_and_final(lambda x,t,c:predict(student,x,t,c),schedule,c,noise,knots,k,both)
            y,final=views[:2]
            if repeat==1:
                disc.set_heads(True);od.zero_grad(set_to_none=True)
                with torch.no_grad():
                    real_rgb=decode_view(vae,torch.randn_like(y),None,kind='whole')
                    negative=bd.gan_refinement(lambda x,t,c:predict(student,x,t,c),schedule,final,c,249,
                                               torch.randn_like(final)) if flash else final
                    fake_rgb=decode_view(vae,negative,None,kind='whole')
                pr=disc(real_rgb,return_parts=True,feature_r1=True);pf=disc(fake_rgb)
                dloss=discriminator_loss(pr['total'],pf)+1e-5*pr['r1']
                dloss.backward();od.step();od.zero_grad(set_to_none=True);disc.set_heads(False);dcount+=1
            of.zero_grad(set_to_none=True)
            u=torch.randint(ranges[k][0],ranges[k][1]+1,(2,));eps=torch.randn_like(y)
            x=bd.renoise(schedule,y,knots[k],u,eps)
            floss=bd.fake_loss(schedule,y,knots[k],u,eps,predict(fake,x,u,c,k))
            if both:
                other=views[2];s=5-k
                u=torch.randint(ranges[s][0],ranges[s][1]+1,(2,));eps=torch.randn_like(other)
                x=bd.renoise(schedule,other,knots[s],u,eps)
                floss=.5*(floss+bd.fake_loss(schedule,other,knots[s],u,eps,predict(fake,x,u,c)))
            floss.backward();fnorm=torch.nn.utils.clip_grad_norm_(fake.parameters(),1,error_if_nonfinite=True)
            of.step();of.zero_grad(set_to_none=True);fcount+=1
        og.zero_grad(set_to_none=True)
        noise,c=batch();mid=bd.middle_stage(g,20261017)
        losses={}
        for k,y in bd.local_rollout(lambda x,t,c:predict(student,x,t,c),schedule,c,noise,knots,(2,3) if flash and not paired_mix else mid):
            dmd=dm(y,k,c);loss=.5*dmd if flash and not paired_mix and k!=4 else dmd;losses[f'dm_{k}']=dmd.item()
            if k==4:
                view=bd.gan_refinement(lambda x,t,c:predict(student,x,t,c),schedule,y,c,249,torch.randn_like(y)) if flash else y
                rgb=decode_view(vae,view,None,kind='whole',grad=True,chunk_size=1)
                gan=generator_loss(disc(rgb));loss=loss+.005*gan;losses['gan']=gan.item()
                if flash:
                    assert torch.autograd.grad(gan,y,allow_unused=True,retain_graph=True)[0] is None
                    assert torch.autograd.grad(gan,view,retain_graph=True)[0].norm().item()>0
            loss.backward()
        real=torch.randn_like(noise);t=torch.randint(770,1000,(2,));eps=torch.randn_like(real)
        xh=schedule.noise(real,eps,t)
        with torch.no_grad():
            xk=xh
            for j in range(2):
                tj=t-10*j;xk=schedule.jump(xk,predict(teacher,xk,tj,c),tj,tj-10)
            target=schedule.jump(xk,predict(student,xk,t-20,c),t-20,749)
        pcm=(schedule.jump(xh,predict(student,xh,t,c),t,749)-target).square().mean()
        pcm.backward();gnorm=torch.nn.utils.clip_grad_norm_(student.parameters(),1,error_if_nonfinite=True)
        og.step();og.zero_grad(set_to_none=True)
        if paired_mix:
            from fake_tracking import mix_shared_parameters
            param=next(fake.parameters()); before=param.detach().clone()
            mix_shared_parameters(fake,student,.01)
            torch.testing.assert_close(param,.99*before+.01*next(student.parameters()))
            assert {key for key in losses if key.startswith('dm_')}=={f'dm_{mid}','dm_4'}
        assert all(p.grad is None for m in (teacher,vae,disc,fake) for p in m.parameters())
        row=dict(G=g+1,middle=mid,fake_updates=fcount,D_updates=dcount,pcm=pcm.item(),
                 fake_loss=floss.item(),R1=pr['r1'].item(),G_grad_norm=gnorm.item(),fake_grad_norm=fnorm.item(),**losses)
        assert all(torch.isfinite(torch.tensor(v)).all() for v in row.values())
        rows.append(row)
    assert (fcount,dcount)==(8,4)
    assert any(not torch.equal(old,new) for old,new in zip(initial_student,student.parameters()))
    return dict(scope='CPU random tiny UNet + VAE + SAM; not pretrained SDXL or FSDP/CUDA validation',
        torch=torch.__version__,G_updates=4,fake_updates=fcount,D_updates=dcount,
        passed=True,wall_seconds=time.monotonic()-begin,rows=rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--flash',action='store_true')
    parser.add_argument('--paired-mix',action='store_true')
    args=parser.parse_args();result=run(args.flash,args.paired_mix)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
