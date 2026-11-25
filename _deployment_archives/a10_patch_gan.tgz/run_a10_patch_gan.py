from pathlib import Path
import fcntl,hashlib,json,math,os,sys
import torch
from diffusers import UNet2DConditionModel
from launch import command,flags
from train import arguments
from queue_dm_pair_coco import read,save,evaluate_checkpoint,reference,timing
from run_dm1_uniform_control import run_gpu
import run_dm1_uniform_control as guard
root=Path(os.environ['SDXL_ROOT'])
runs=root/'outputs';job=runs/'A_pcm10_terminal_patchgan'; code=Path(__file__).resolve().parent
guard.out=job
with (job/'queue.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 parent=runs/'A_pcm10_g160'
 cfg=read(parent/'effective_config.json')
 assert cfg['pcm_weight']==10 and not cfg['turning_profile'] and cfg['fake_updates']==2 and cfg['gan_weight']==0
 cfg.update(variant='joint4',gan_weight=.005274229682981968,discriminator='hybrid',gan_view='tile',gan_augment='d_only')
 sys.argv=['train','--root',str(root),'--run',job.name+'/model',*flags(cfg)]
 args=arguments()
 assert args.batch==64 and args.max_student_updates==160 and args.pair_fake_stages
 save(job/'effective_config.json',cfg)
 model=job/'model'
 if not (model/'status.json').exists():
    if (job/'train.log').exists():raise RuntimeError('Inspect previous attempt')
    run_gpu(command(cfg,root,job.name+'/model',1),job/'train.log')
 status=read(model/'status.json')
 assert (status['status'],status['student_updates'],status['fake_updates'],status['D_updates'])==('complete',160,320,800)
 entries=[json.loads(l) for l in (model/'rank0.jsonl').read_text().splitlines()]
 gs=[r for r in entries if r['phase']=='student_done'];fs=[r for r in entries if r['phase']=='fake'];ds=[r for r in entries if r['phase']=='D']
 assert [g['g'] for g in gs if g['gan']]==list(range(3,161,3))
 assert [r['D_update'] for r in ds]==list(range(1,801)) and sum(r['R1_active'] for r in ds)==50
 assert len(fs)==320 and len(gs)==160 and all([r['fake_stage'] for r in fs[i*2:i*2+2]]==[g['stage'],4] for i,g in enumerate(gs))
 assert all(math.isfinite(g[k]) for g in gs for k in ('pcm_loss','gan_loss','dmd_loss','grad_norm'))
 ck,=[p.parent for p in (model/'weights').glob('*/complete.json')]
 state=torch.load(ck/'student_rank0.pt',map_location='cpu',mmap=True,weights_only=True)
 with torch.device('meta'): net=UNet2DConditionModel.from_config(read(root/'models/sdxl-base-1.0/unet/config.json'))
 net.load_state_dict(state,strict=True,assign=True)
 assert all(torch.isfinite(v).all() for v in state.values())
 del net,state
 save(ck/'cpu_validation.json',dict(strict_load=True,finite=True))
 scores=evaluate_checkpoint('A_pcm10_patchgan',ck,read(model/'config.json'),read(reference/'manifest.json'),job,run_worker=run_gpu)
 base=read(parent/'comparison.json')['A_pcm10']
 save(job/'comparison.json',dict(A_pcm10=base,A_pcm10_patchgan=dict(**scores,**timing(model)),delta={k:v-base[k] for k,v in scores.items()}))

