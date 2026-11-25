from pathlib import Path
import torch
from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');pilot=root/'outputs/hparam_pilot';out=pilot/'early_preview';out.mkdir(exist_ok=True)
    vae=AutoencoderKL.from_pretrained(root/'models/sdxl-base-1.0',subfolder='vae',variant='fp16',torch_dtype=torch.float32,local_files_only=True).eval()
    processor=VaeImageProcessor(vae_scale_factor=8)
    initial=torch.load(pilot/'lr5e7_pcm10/validation_before_rank0.pt',weights_only=True)[0]
    trained=torch.load(pilot/'lr5e7_pcm10/validation_after_rank0.pt',weights_only=True)[0]
    teacher=torch.load(pilot/'validation_refs/tau749_rank0.pt',weights_only=True)[0]
    for name,row in [('untrained',initial),('trained',trained),('teacher',teacher)]:
        decoded=vae.decode(row['final'].float()/vae.config.scaling_factor,return_dict=False)[0]
        processor.postprocess(decoded,output_type='pil')[0].save(out/f'{name}.png')
        print('PREVIEW_SAVED',name,flush=True)
    (out/'prompt.txt').write_text(initial['prompt'])
    print('EARLY_PREVIEW_COMPLETE',flush=True)

if __name__=='__main__':main()
