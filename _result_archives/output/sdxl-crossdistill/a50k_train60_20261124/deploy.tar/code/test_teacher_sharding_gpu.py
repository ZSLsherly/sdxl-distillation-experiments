"""Actual frozen SDXL CFG output equivalence, one identical path per rank."""
import argparse,gc,json,os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._composable.fsdp import MixedPrecisionPolicy
from diffusers import UNet2DConditionModel
from batching import conditions,cfg_prediction
from train import shard_unet


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']));device=torch.device('cuda',int(os.environ['LOCAL_RANK']))
    torch.set_num_threads(2);dist.init_process_group('nccl');mesh=init_device_mesh('cuda',(world,))
    data=torch.load(args.root/'datasets/parallel_10k_seed20261018/prepared.pt',weights_only=True,mmap=True,map_location='cpu')
    cond=conditions(data,[rank],device)
    x=torch.randn((1,4,128,128),device=device,generator=torch.Generator(device=device).manual_seed(800+rank))
    def load():
        return UNet2DConditionModel.from_pretrained(args.root/'models/sdxl-base-1.0/unet',variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True).to(device).eval().requires_grad_(False)
    def predict(model,x,t,c,negative=False):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            return model(x,t,encoder_hidden_states=c['negative_prompt_embeds' if negative else 'prompt_embeds'],
                         added_cond_kwargs={'text_embeds':c['negative_pooled_prompt_embeds' if negative else 'pooled_prompt_embeds'],
                                            'time_ids':c['time_ids']},return_dict=False)[0].float()
    times=[777,249];reference=[]
    with torch.no_grad():
        teacher=load()
        for value in times:
            t=torch.tensor([value],device=device,dtype=torch.long)
            reference.append(cfg_prediction(predict,teacher,x,t,cond,8.,False).cpu())
        del teacher;gc.collect();torch.cuda.empty_cache();dist.barrier()
        teacher=load()
        policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16,reduce_dtype=torch.float32,cast_forward_inputs=False)
        shard_unet(teacher,mesh,policy,'fine')
        rows=[]
        for value,expected in zip(times,reference):
            actual=cfg_prediction(predict,teacher,x,torch.tensor([value],device=device,dtype=torch.long),cond,8.,False).cpu()
            torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-5)
            rows.append({'t':value,'max_abs_difference':(actual-expected).abs().max().item(),
                         'relative_L2':((actual-expected).norm()/expected.norm().clamp_min(1e-12)).item()})
        assert all(not p.requires_grad and p.grad is None for p in teacher.parameters())
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/f'rank{rank}.json').write_text(json.dumps({'rank':rank,'passed':True,'rows':rows},indent=2))
    print('TEACHER_EQUIVALENCE_PASSED',rank,rows,flush=True)
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
