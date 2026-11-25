"""Frozen G160 student: one teacher interval replacement, unchanged student tail."""
import argparse,gc,json,sys,time,hashlib
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from diffusers import UNet2DConditionModel,DDIMScheduler,AutoencoderKL
from transformers import Dinov2Model
from stage_intervention_core import integrate,student_rollout,replace_stage

VARIANTS=['student','replace1','replace2','replace3','replace4','teacher']
KNOTS=[999,749,499,249,-1]

def load_student(model,checkpoint):
    meta=json.loads((checkpoint/'shapes_rank0.json').read_text())
    shards=[torch.load(checkpoint/f'student_rank{i}.pt',weights_only=True,map_location='cpu',mmap=True) for i in range(2)]
    state={}
    for k,v in meta.items():
        if v['placements'] and v['placements'][0].startswith('S('):
            state[k]=torch.cat([s[k] for s in shards],dim=int(v['placements'][0][2:-1]))
        else:
            assert torch.equal(shards[0][k],shards[1][k]),k
            state[k]=shards[0][k]
        assert list(state[k].shape)==v['shape']
    model.load_state_dict(state,strict=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--rank',type=int,default=0);p.add_argument('--workers',type=int,default=1)
    p.add_argument('--batch',type=int,default=4);p.add_argument('--limit',type=int,default=32);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(2);torch.cuda.set_device(0)
    torch.cuda.set_per_process_memory_fraction(.65)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    def cooperate():
        while (a.out/f'PAUSE_{a.rank}').exists():time.sleep(1)
    base=a.root/'models/sdxl-base-1.0';assets=a.root/'stage_assets';checkpoint=assets/'checkpoint'
    cfg=json.loads((checkpoint/'config.json').read_text());assert cfg['schedule']==KNOTS and cfg['teacher_cfg']==8
    assert json.loads((checkpoint/'complete.json').read_text())['student_updates']==160
    d=torch.load(a.root/'conditions512.pt',weights_only=True,map_location='cpu')
    ids=list(range(96,96+a.limit))[a.rank::a.workers]
    sched=DDIMScheduler.from_pretrained(base,subfolder='scheduler',local_files_only=True)
    alpha=sched.alphas_cumprod.sqrt().cuda();sigma=(1-sched.alphas_cumprod).sqrt().cuda()
    def jump(x,eps,t,s):
        clean=(x-sigma[t]*eps)/alpha[t]
        return clean if s<0 else alpha[s]*clean+sigma[s]*eps
    student=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False)
    load_student(student,checkpoint);student.cuda();gc.collect()
    teacher=UNet2DConditionModel.from_pretrained(base,subfolder='unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).eval().requires_grad_(False).cuda()
    begin=time.monotonic();torch.cuda.reset_peak_memory_stats();checks=[]
    states=outputs=noop=x=y=boundary=None
    with torch.inference_mode():
        for offset in range(0,len(ids),a.batch):
            group=ids[offset:offset+a.batch]
            if all((a.out/f'latent_{i:03d}.pt').exists() for i in group):continue
            b=len(group)
            def cond(key):
                v=d[key];return (v.expand(b,*v.shape[1:]) if len(v)==1 else v[group]).to('cuda',dtype=torch.bfloat16)
            pe,pooled,tids=cond('prompt_embeds'),cond('pooled_prompt_embeds'),cond('time_ids')
            tpe=torch.cat([cond('negative_prompt_embeds'),pe]);tpp=torch.cat([cond('negative_pooled_prompt_embeds'),pooled]);tti=tids.repeat(2,1)
            def predict_s(x,t):
                cooperate()
                return student(x.to(torch.bfloat16),torch.tensor([t],device='cuda'),encoder_hidden_states=pe,
                    added_cond_kwargs={'text_embeds':pooled,'time_ids':tids},return_dict=False)[0].float()
            def predict_t(x,t):
                cooperate()
                values=teacher(torch.cat([x,x]).to(torch.bfloat16),torch.tensor([t],device='cuda'),encoder_hidden_states=tpe,
                    added_cond_kwargs={'text_embeds':tpp,'time_ids':tti},return_dict=False)[0].float()
                uncond,cond_eps=values.chunk(2);return uncond+8*(cond_eps-uncond)
            x=torch.cat([torch.randn((1,4,128,128),device='cuda',generator=torch.Generator(device='cuda').manual_seed(201)) for i in group])
            states=student_rollout(x,KNOTS,predict_s,jump);outputs={'student':states[-1]};local=[]
            # Exact no-op control: recompute stage 2 and its tail with the SAME student.
            noop=student_rollout(states[1].clone(),KNOTS,predict_s,jump,start_stage=1)[-1]
            max_error=(noop-states[-1]).abs().max().item();assert max_error==0,max_error
            checks.append({'ids':group,'noop_max_abs':max_error})
            for stage in range(4):
                y,boundary=replace_stage(states,stage,KNOTS,predict_s,predict_t,jump)
                outputs[f'replace{stage+1}']=y
                local.append((boundary-states[stage+1]).square().flatten(1).mean(1).sqrt().cpu().tolist())
                print('STAGE_DONE',a.rank,group,stage+1,flush=True)
            outputs['teacher']=integrate(x.clone(),999,-1,predict_t,jump)
            for j,i in enumerate(group):
                values={k:v[j:j+1].cpu() for k,v in outputs.items()}
                assert all(torch.isfinite(v).all() for v in values.values())
                torch.save(values,a.out/f'latent_{i:03d}.pt')
                (a.out/f'case_{i:03d}.json').write_text(json.dumps({'prompt_index':i,'source_index':d['source_indices'][i],
                    'prompt':d['prompts'][i],'seed':201,'local_boundary_RMS_by_stage':[v[j] for v in local],
                    'student_cfg':1,'teacher_cfg':8,'knots':KNOTS,'teacher_stride':10,'sampling':'deterministic DDIM; no re-noise',
                    'student_checkpoint':'uniform joint4 G160','rank':a.rank},ensure_ascii=False,indent=2))
            print('BATCH_DONE',a.rank,group,round(time.monotonic()-begin,2),flush=True)
    peak=torch.cuda.max_memory_allocated()/2**30
    del student,teacher,states,outputs,noop,x,y,boundary
    gc.collect();torch.cuda.empty_cache()
    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval().requires_grad_(False).cuda()
    with torch.inference_mode():
        for i in ids:
            values=torch.load(a.out/f'latent_{i:03d}.pt',weights_only=True,map_location='cpu')
            for name in VARIANTS:
                path=a.out/f'{i:03d}_{name}.png'
                if path.exists():continue
                cooperate();rgb=vae.decode(values[name].cuda()/vae.config.scaling_factor,return_dict=False)[0]
                assert torch.isfinite(rgb).all()
                pixels=((rgb[0]/2+.5).clamp(0,1).permute(1,2,0)*255).round().byte().cpu().numpy()
                Image.fromarray(pixels).save(path)
            print('DECODE_DONE',a.rank,i,flush=True)
    del vae;gc.collect();torch.cuda.empty_cache()
    sys.path.insert(0,str(assets/'packages'));import lpips
    percept=lpips.LPIPS(net='squeeze',pnet_rand=True,pretrained=False,verbose=False).eval().requires_grad_(False)
    percept.load_state_dict(torch.load(assets/'lpips_squeeze.pt',weights_only=True,map_location='cpu'),strict=True);percept.cuda()
    dino=Dinov2Model.from_pretrained(assets/'dinov2-small',local_files_only=True).eval().requires_grad_(False).cuda()
    mean=torch.tensor([.485,.456,.406],device='cuda')[None,:,None,None];std=torch.tensor([.229,.224,.225],device='cuda')[None,:,None,None]
    rows=[]
    with torch.inference_mode():
        for i in ids:
            cooperate()
            im=torch.stack([torch.from_numpy(np.asarray(Image.open(a.out/f'{i:03d}_{n}.png').convert('RGB')).copy()).permute(2,0,1) for n in VARIANTS]).float().cuda()/255
            small=F.interpolate(im,size=(224,224),mode='bicubic',align_corners=False,antialias=True)
            features=F.normalize(dino(pixel_values=(small-mean)/std).last_hidden_state[:,0].float(),dim=1)
            full=F.interpolate(im,size=(512,512),mode='bilinear',align_corners=False,antialias=True)*2-1
            patches=torch.stack([im[:,:,yy:yy+256,xx:xx+256] for yy,xx in [(128,128),(128,640),(640,128),(640,640)]],dim=1)*2-1
            low=F.interpolate(im,size=(32,32),mode='area')
            hp=im[:,:,1:-1,1:-1]*4-im[:,:,:-2,1:-1]-im[:,:,2:,1:-1]-im[:,:,1:-1,:-2]-im[:,:,1:-1,2:]
            energy=hp.square().mean((1,2,3)).sqrt()
            for j,name in enumerate(VARIANTS):
                rec={'prompt_index':i,'variant':name,'highpass_RMS':energy[j].item()}
                for ref,label in [(0,'student'),(5,'teacher')]:
                    rec[f'dino_distance_to_{label}']=(1-(features[j]*features[ref]).sum()).clamp_min(0).item()
                    rec[f'lpips_full_to_{label}']=percept(full[j:j+1],full[ref:ref+1]).item()
                    rec[f'lpips_patch_to_{label}']=percept(patches[j],patches[ref]).mean().item()
                    rec[f'lowfreq_L1_to_{label}']=(low[j]-low[ref]).abs().mean().item()
                    rec[f'highpass_logratio_to_{label}']=((energy[j]+1e-8)/(energy[ref]+1e-8)).log().abs().item()
                assert all(np.isfinite(v) for v in rec.values() if isinstance(v,float))
                rows.append(rec)
            print('METRICS_DONE',a.rank,i,flush=True)
    (a.out/f'metrics_rank{a.rank}.json').write_text(json.dumps(rows,indent=2))
    (a.out/f'worker{a.rank}_complete.json').write_text(json.dumps({'complete':True,'count':len(ids),'peak_sampling_GiB':peak,
        'seconds_after_model_load':time.monotonic()-begin,'checks':checks},indent=2))
    print('WORKER_COMPLETE',a.rank,len(ids),flush=True)

if __name__=='__main__':main()
