from pathlib import Path
import shutil,json
base=Path('output/sdxl-crossdistill/dmd2_matched_20261119')
dst=Path('output/sdxl-crossdistill/dmd2_official_recipe_20261121')
dst.mkdir(exist_ok=True)
if not (dst/'code').exists():shutil.copytree(base/'code',dst/'code',ignore=shutil.ignore_patterns('__pycache__'))
cfg=json.loads((base/'config.json').read_text(encoding='utf-8'))
cfg.update(name='DMD2_official_optimizer_precision_10K_G160_F800',student_lr=5e-7,fake_lr=5e-7,
 betas=[.9,.999],weight_decay=.01,grad_clip=10.,warmup_scheduler_steps=500,warmup_reference_processes=64,
 gpus=[1,3],gpu_uuids=['GPU-dee80cca-d29a-9086-d006-b76ba6ee093e','GPU-c374184a-ab5a-681a-ed48-025bed2da39b'],
 runtime='FSDP2 FULL_SHARD; student/fake/head FP32 parameters and optimizer; upstream selective BF16 autocast; teacher BF16; no CPU offload',
 evaluation='official stochastic four-step sampler; eight fixed cases at G0/every2G; full COCO10K comparison after G160',
 warmup_semantics='lr(iter)=base_lr*min(iter*64/500,1); reproduce official 64-process Accelerate0.23 trajectory independently of this job GPU count',
 retained_differences='10K cached dataset, batch64, G160/F800, FP16-source initialization, FSDP2/checkpointing and student DM upper979 retained from prior run')
(dst/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2),encoding='utf-8')
p=dst/'code/train_matched.py';s=(base/'code/train_matched.py').read_text(encoding='utf-8')
s=s.replace("assert world==4 and 64%(world*a.micro)==0","assert world in (2,4) and 64%(world*a.micro)==0")
s=s.replace("policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=True)","# Keep FP32 fake-score/rollout; upstream owns selective autocast.\n            policy=MixedPrecisionPolicy(param_dtype=None,reduce_dtype=torch.float32,cast_forward_inputs=False)")
s=s.replace("adam=dict(betas=(0.,.999),weight_decay=0.,foreach=False)","adam=dict(betas=tuple(cfg['betas']),weight_decay=cfg['weight_decay'],foreach=False)")
s=s.replace("og=torch.optim.AdamW(student_params,lr=1e-6,**adam);of=torch.optim.AdamW(fake_params,lr=5e-6,**adam)","og=torch.optim.AdamW(student_params,lr=cfg['student_lr'],**adam);of=torch.optim.AdamW(fake_params,lr=cfg['fake_lr'],**adam)")
s=s.replace("    @torch.no_grad()\n    def evaluate(update):",'''    precision_seen={}
    def precision_hook(role):
        def hook(module, inputs, output):
            amp=torch.is_autocast_enabled()
            key=f'{role}:autocast={amp}:grad={torch.is_grad_enabled()}'
            value=dict(input=str(inputs[0].dtype),weight=str(module.weight.dtype),output=str(output.dtype))
            if role in ('student','fake'):
                assert module.weight.dtype==torch.float32,(key,value)
                if not amp:assert inputs[0].dtype==output.dtype==torch.float32,(key,value)
            elif role=='teacher':assert module.weight.dtype==torch.bfloat16,(key,value)
            if key not in precision_seen:
                precision_seen[key]=value
                write(out/f'precision_rank{rank}.json',precision_seen)
        return hook
    for role,net in [('student',model.feedforward_model),('fake',model.guidance_model.fake_unet),('teacher',model.guidance_model.real_unet)]:
        net.conv_in.register_forward_hook(precision_hook(role))
    @torch.no_grad()
    def evaluate(update):''')
s=s.replace("                x=torch.randn(1,4,128,128,device=device,generator=torch.Generator(device=device).manual_seed(102))", "                torch.manual_seed(102)\n                x=torch.randn(1,4,128,128,device=device)")
s=s.replace("                    x=schedule.jump(x,eps,tt,s)","                    x0=schedule.clean(x,eps,tt)\n                    x=schedule.noise(x0,torch.randn_like(x0),s) if s>=0 else x0")
s=s.replace("sampler='shared historical deterministic DDIM4; NOT training backward simulation'", "sampler='official DMD2 x0 prediction + independent Gaussian renoise; four-step; BF16 autocast evaluation'")
s=s.replace("        tick=time.monotonic();do_g=(iteration%5==0)","        tick=time.monotonic();do_g=(iteration%5==0)\n        factor=min(iteration*cfg['warmup_reference_processes']/cfg['warmup_scheduler_steps'],1.)\n        for group in og.param_groups:group['lr']=cfg['student_lr']*factor\n        for group in of.param_groups:group['lr']=cfg['fake_lr']*factor")
s=s.replace("gn=clip(student_params,1.);og.step();g+=1", "gn=clip(student_params,cfg['grad_clip']);og.step();g+=1")
s=s.replace("fn=clip(fake_params,1.);of.step();", "fn=clip(fake_params,cfg['grad_clip']);of.step();")
s=s.replace("record(dict(iteration=iteration,G=g,F=f,student_updated=do_g,depth=selected,", "record(dict(iteration=iteration,G=g,F=f,student_updated=do_g,depth=selected,\n                    student_lr=og.param_groups[0]['lr'],fake_lr=of.param_groups[0]['lr'],")
p.write_text(s,encoding='utf-8')
# Separate root and logs; old training/code/checkpoints remain untouched.
p=dst/'code/run_guard.py';s=(base/'code/run_guard.py').read_text(encoding='utf-8')
s=s.replace("'--nproc_per_node=4'","f\"--nproc_per_node={len(cfg['gpu_uuids'])}\"")
s=s.replace("(16,8,4,2)","(4,2,1)")
s=s.replace("range(4)","range(len(cfg['gpu_uuids']))")
s=s.replace("==2\n            selected", "==8//len(cfg['gpu_uuids'])\n            selected")
s=s.replace("64//(4*micro)","64//(len(cfg['gpu_uuids'])*micro)")
# Short test covers full G and fake updates, four rollout depths, save/load and eval.
s=s.replace("            selected=micro\n",'''            for rank in range(len(cfg['gpu_uuids'])):
                precision=json.loads((r/'runs'/name/f'precision_rank{rank}.json').read_text())
                assert precision['fake:autocast=False:grad=False']['output']=='torch.float32'
                assert precision['student:autocast=False:grad=False']['output']=='torch.float32'
                assert precision['teacher:autocast=False:grad=False']['weight']=='torch.bfloat16'
                rows=[json.loads(x) for x in (r/'runs'/name/f'rank{rank}.jsonl').read_text().splitlines()]
                assert rows[0]['student_lr']==0 and rows[5]['student_lr']==cfg['student_lr']*.64
                assert rows[8]['fake_lr']==cfg['fake_lr']
            selected=micro
''')
p.write_text(s,encoding='utf-8')
print(dst.resolve())
