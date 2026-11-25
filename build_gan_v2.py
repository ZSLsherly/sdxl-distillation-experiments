"""Build a versioned v2 trainer, retaining the reviewed v1 sources untouched."""
from pathlib import Path
p=Path('sdxl_gan_train.py');s=p.read_text(encoding='utf-8')
def change(old,new):
    global s
    assert old in s,old[:100]
    s=s.replace(old,new)
change('from sdxl_pixel_gan import PixelPatchDiscriminator,decode_tile,discriminator_loss,average_gradients','from sdxl_pixel_gan import PixelPatchDiscriminator,decode_tile,discriminator_loss,average_gradients\nfrom sdxl_gan_policy_v2 import VARIANTS,gan_active,dmd_active,discriminator_stage,variant_for_iteration,check_policy')
change("choices=['dmd','gan4','joint4','joint34']","choices=VARIANTS")
change("    ap.add_argument('--calibrate-only',action='store_true')","    ap.add_argument('--gan-weight',type=float,default=0.003530974732711911)\n    ap.add_argument('--series',default='gan_stage_pilot_v2')")
change("    args=ap.parse_args()","    args=ap.parse_args();check_policy()\n    assert args.student_updates>0\n    if not args.preflight:assert args.student_updates%6==0,'Matched exposure requires complete two-cycle blocks'")
change("    assert not (out/'summary.json').exists(),'Completed run cannot be overwritten'","    assert not (out/'rank0.jsonl').exists(),'Run directory already contains updates; use a new run id'")
change("for src in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py')]:","for src in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py'),Path(__file__).with_name('sdxl_gan_policy_v2.py')]:")
change("for p in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py')]","for p in [Path(__file__),Path(__file__).with_name('sdxl_pixel_gan.py'),Path(__file__).with_name('sdxl_gan_policy_v2.py')]")
change("    calibration=root/'outputs/gan_stage_pilot/calibration'","    series=root/'outputs'/args.series;series.mkdir(parents=True,exist_ok=True)\n    has_discriminator=args.variant!='dmd' or args.preflight\n    gan_weight=args.gan_weight\n    def sync_time():\n        torch.cuda.synchronize();return time.perf_counter()\n    def write_json(path,value):\n        temporary=path.with_suffix(path.suffix+'.tmp');temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)")
change("    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to(device).eval().requires_grad_(False)\n    discriminator=PixelPatchDiscriminator().to(device)","    vae=AutoencoderKL.from_pretrained(base,subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).to(device).eval().requires_grad_(False) if has_discriminator else None\n    torch.manual_seed(args.seed+7000)\n    discriminator=PixelPatchDiscriminator().to(device).eval().requires_grad_(False) if has_discriminator else None")
change("    opt_d=torch.optim.AdamW(discriminator.parameters(),lr=1e-4,betas=(0,.99),weight_decay=0)\n    if not args.calibrate_only:\n        discriminator.load_state_dict(torch.load(calibration/'discriminator.pt',weights_only=True,map_location='cpu'))\n        gan_weight=json.loads((calibration/'summary.json').read_text())['gan_weight']","    opt_d=torch.optim.AdamW(discriminator.parameters(),lr=1e-4,betas=(0,.99),weight_decay=0) if has_discriminator else None")
change("def rollout(c,grad=False,stage=4,noise=None):","def rollout(c,grad=False,stage=4,noise=None,return_clean_predictions=False):")
change("        for i in range(stage):","        clean_predictions={}\n        for i in range(stage):")
change("                x=sched.jump(x,predict(student,x,tt,c),tt,s)","                epsilon=predict(student,x,tt,c)\n                if return_clean_predictions:clean_predictions[i+1]=sched.clean(x,epsilon,tt).detach()\n                x=sched.jump(x,epsilon,tt,s)")
change("        return x\n    def grad_norm", "        return (x,clean_predictions) if return_clean_predictions else x\n    def grad_norm")
change("        discriminator.requires_grad_(True).train();opt_d.zero_grad(set_to_none=True)","        start_decode=sync_time()\n        discriminator.requires_grad_(True).train();opt_d.zero_grad(set_to_none=True)")
change("        flip=number%2==0","        end_decode=sync_time()\n        flip=number%2==0")
change("return {'discriminator_loss':loss.item(),'real_logit':lr.mean().item(),'fake_logit':lf.mean().item(),'r1':r1.item()}","return {'discriminator_loss':loss.item(),'real_logit':lr.mean().item(),'fake_logit':lf.mean().item(),'r1':r1.item(),'r1_active':r1_on,'d_decode_seconds':end_decode-start_decode,'d_update_seconds':sync_time()-end_decode}")
start=s.index('    if args.calibrate_only:');end=s.index('    @torch.no_grad()',start)
s=s[:start]+s[end:]
change("refs_dir=root/'outputs/gan_stage_pilot/validation_refs_112_119'","refs_dir=series/'validation_refs_112_119'")
change("    for iteration in range(args.student_updates*6):","    stopped=False;d_updates=0\n    for iteration in range(args.student_updates*6):")
change("        is_g=iteration%6==5;g_step=iteration//6","        is_g=iteration%6==5;g_step=iteration//6\n        active_variant=variant_for_iteration(args.variant,g_step,args.preflight)")
start=s.index('        extras={}\n        if not is_g:');end=s.index('        else:\n            stage=2+g_step%3;',start)
s=s[:start]+'''        extras={'active_variant':active_variant,'d_fake_stage':None,'d_decode_seconds':0.0,'d_update_seconds':0.0,'fake_score_seconds':0.0}
        if not is_g:
            begin_rollout=sync_time()
            with torch.no_grad():generated,clean_predictions=rollout(c,return_clean_predictions=True)
            extras['rollout_seconds']=sync_time()-begin_rollout
            d_stage=discriminator_stage(active_variant,d_updates)
            if d_stage is not None:
                extras.update(d_step(clean_predictions[d_stage],real,c,xy,d_updates))
                extras['d_fake_stage']=d_stage;d_updates+=1
            del clean_predictions
            begin_fake=sync_time()
            t=torch.randint(0,1000,(1,),device=device);eps=torch.randn_like(generated);xt=sched.noise(generated,eps,t)
            loss=(predict(fake,xt,t,c)-eps).square().mean();loss.backward();norm,clip=grad_norm(fake);opt_f.step();phase='fake_score'
            extras['fake_score_seconds']=sync_time()-begin_fake
            assert all(p.grad is None for p in student.parameters())
''' + s[end:]
change("            stage=2+g_step%3;generated=rollout(c,True,stage)","            stage=2+g_step%3;begin_rollout=sync_time();generated=rollout(c,True,stage)\n            extras['rollout_seconds']=sync_time()-begin_rollout\n            low_start=sync_time()")
change("            use_dmd=not(args.variant=='gan4' and stage==4)\n            use_gan=(stage==4 and args.variant!='dmd') or (stage==3 and args.variant=='joint34')","            use_dmd=dmd_active(active_variant,g_step)\n            use_gan=gan_active(active_variant,g_step)")
change("loss=torch.zeros((),device=device);extras={'generator_stage':stage,'dmd_active':use_dmd,'gan_active':use_gan}","loss=torch.zeros((),device=device);extras.update(generator_stage=stage,dmd_active=use_dmd,gan_active=use_gan)")
change("                rgb=decode_tile(vae,generated,xy,True)","                gan_start=sync_time();rgb=decode_tile(vae,generated,xy,True)\n                extras['g_gan_decode_forward_seconds']=sync_time()-gan_start")
change("            loss.backward();low_loss=loss.item()","            loss.backward();low_loss=loss.item()\n            extras['low_objective_and_backward_seconds']=sync_time()-low_start\n            pcm_start=sync_time()")
change("            norm,clip=grad_norm(student);opt_g.step();phase='student'","            norm,clip=grad_norm(student);opt_g.step();phase='student'\n            extras['pcm_and_optimizer_seconds']=sync_time()-pcm_start")
change("            assert all(p.grad is None for p in discriminator.parameters())","            assert discriminator is None or all(p.grad is None for p in discriminator.parameters())")
change("assert all(p.grad is None for p in teacher.parameters()) and all(p.grad is None for p in vae.parameters())","assert all(p.grad is None for p in teacher.parameters()) and (vae is None or all(p.grad is None for p in vae.parameters()))")
needle="        print(f'RANK {rank}: STEP {iteration} {phase} {elapsed:.2f}s loss={loss.item():.6g} peak={row[\"peak_allocated_GiB\"]:.2f}GiB',flush=True)"
change(needle,needle+'''
        stop_flag=torch.tensor(int((series/'STOP_REQUEST').exists()),device=device)
        dist.all_reduce(stop_flag,op=dist.ReduceOp.MAX)
        if stop_flag.item():stopped=True;break''')
change("    if not args.preflight:validate('after')","    if not args.preflight and not stopped:validate('after')")
change(".abs().max().item();assert delta>0",".abs().max().item()\n    assert stopped or delta>0")
change("'status':'training_complete'","'status':'stopped_checkpointed' if stopped else ('preflight_complete' if args.preflight else 'training_complete')")
change("'student_updates':args.student_updates,'fake_updates':args.student_updates*5,'pixel_discriminator_updates':args.student_updates*5","'student_updates':sum(r['phase']=='student' for r in records),'fake_updates':sum(r['phase']=='fake_score' for r in records),'pixel_discriminator_updates':d_updates")
change("'gan_weight_method':'common initial RGB-to-latent gradient calibration, target 0.15 DMD, fixed thereafter'","'gan_weight_method':'fixed exploratory coefficient inherited from v1 magnitude probe; fresh common random D initialization; not an optimal weight or a maintained 15% ratio'")
change("'seed':args.seed,'EMA':False}","'seed':args.seed,'EMA':False,'revision':'v2','D_negative_stages':'none' if args.variant=='dmd' else ('equal mixture 3/4' if args.variant.startswith('joint34') else 'stage 4'),'GAN_exposure_matching':'joint34_matched uses 4 stage-3 plus 4 stage-4 G updates per 24 updates','checkpoint_kind':'model-only student shards; no optimizer resume'}")
change("    (out/f'summary_rank{rank}.json').write_text(json.dumps(summary,indent=2))","    summary['preflight_extra_R1_at_first_D_update']=bool(args.preflight)\n    summary['D_stage_counts']={str(k):sum(r.get('d_fake_stage')==k for r in records) for k in (3,4)}\n    summary['GAN_G_stage_counts']={str(k):sum(r.get('gan_active',False) and r.get('generator_stage')==k for r in records) for k in (3,4)}")
change("    else:\n        summary['status']='preflight_complete'\n        summary['preflight_extra_R1_at_first_D_update']=True\n    dist.barrier();stop.set()","    dist.barrier();stop.set()\n    if rank==0 and not args.preflight and discriminator is not None:torch.save(discriminator.state_dict(),out/'discriminator.pt')\n    dist.barrier()\n    write_json(out/f'summary_rank{rank}.json',summary)\n    dist.barrier()")
change("        if not args.preflight:torch.save(discriminator.state_dict(),out/'discriminator.pt')\n",'')
change("        (out/'summary.json').write_text(json.dumps(summary,indent=2))","        write_json(out/'summary.json',summary)")
change("    log.close();print(f'RANK {rank}: TRAIN_COMPLETE',flush=True);dist.destroy_process_group()","    log.close();print(f'RANK {rank}: {summary[\"status\"].upper()}',flush=True);dist.destroy_process_group()\n    if stopped:raise SystemExit(75)")
Path('sdxl_gan_train_v2.py').write_text(s,encoding='utf-8')
print('WROTE sdxl_gan_train_v2.py')
