"""Reconstruct the BF16 inference UNet from model-only FSDP2 rank shards."""
import json,torch,shutil,argparse
from pathlib import Path
from safetensors.torch import save_file
ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--force',action='store_true');args=ap.parse_args()
root=Path('/data5/shenglin.zeng/sdxl_crossdistill');out=root/'outputs'/args.run
world=json.loads((out/'summary.json').read_text())['world_size']
if out.name!='lr5e7_pcm10' and not args.force:
    paths=[out/f'student_rank{rank}.pt' for rank in range(world)]
    assert all(p.is_file() for p in paths)
    (out/'export_deferred.json').write_text(json.dumps({'status':'model-only FSDP shards retained; portable export deferred until selection','shards':[str(p) for p in paths],'export_command':f'python src/sdxl_hparam_export.py --run {args.run} --force'},indent=2))
    print('EXPORT_DEFERRED_CHECKPOINTS_RETAINED',flush=True)
    raise SystemExit(0)
meta=json.loads((out/'student_rank0_shapes.json').read_text())
shards=[torch.load(out/f'student_rank{rank}.pt',map_location='cpu',weights_only=True) for rank in range(world)]
full={}
for k,m in meta.items():
    placements=m['placements']
    if placements and placements[0].startswith('S('):
        dim=int(placements[0][2:-1]);value=torch.cat([s.pop(k) for s in shards],dim=dim)
    else:
        value=shards[0].pop(k)
        for s in shards[1:]:assert torch.equal(value,s.pop(k)),k
    assert list(value.shape)==m['shape'],(k,value.shape,m)
    full[k]=value.contiguous()
dest=out/'export/unet';dest.mkdir(parents=True,exist_ok=True)
save_file(full,dest/'diffusion_pytorch_model.safetensors',metadata={'format':'pt'})
shutil.copyfile(root/'models/sdxl-base-1.0/unet/config.json',dest/'config.json')
print('EXPORTED',len(full),'tensors',sum(v.numel() for v in full.values()),flush=True)
