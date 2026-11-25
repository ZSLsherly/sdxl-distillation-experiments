"""Cache real LAION latents and SDXL's two frozen CLIP encoders for the smoke run."""
import io,pickle,json,hashlib,zipfile,os
from pathlib import Path
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

class DataUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allow={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype,
               ('numpy.core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,
               ('numpy._core.multiarray','_reconstruct'):np.core.multiarray._reconstruct,
               ('numpy.core.numeric','_frombuffer'):np.core.numeric._frombuffer,
               ('numpy._core.numeric','_frombuffer'):np.core.numeric._frombuffer}
        if (module,name) not in allow:raise pickle.UnpicklingError('Unsupported data global '+module+'.'+name)
        return allow[(module,name)]

def load_numpy_pt(path):
    with zipfile.ZipFile(path) as archive:
        name=next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        return DataUnpickler(io.BytesIO(archive.read(name))).load()

def main():
    root=Path('/data5/shenglin.zeng/sdxl_crossdistill');base=root/'models/sdxl-base-1.0'
    original=load_numpy_pt(root/'datasets/dmd2/vae_latents_000.pt')
    print('ORIGINAL_DATA',{k:(type(v).__name__,getattr(v,'shape',None)) for k,v in original.items()},flush=True)
    assert set(original)>={'latents','prompts'}
    indices=np.random.default_rng(20261017).permutation(len(original['latents']))[:128]
    latents=torch.from_numpy(original['latents'][indices].copy())
    assert latents.shape[1:]==(4,128,128),latents.shape
    assert torch.isfinite(latents).all()
    pcm_prompts=[str(original['prompts'][i]) for i in indices]
    all_dmd_prompts=DataUnpickler(io.BytesIO((root/'datasets/dmd2/captions_laion_score6.25.pkl').read_bytes())).load()
    dmd_indices=np.random.default_rng(20261018).permutation(len(all_dmd_prompts))[:128]
    prompts=[str(all_dmd_prompts[i]) for i in dmd_indices]
    eval_prompts=DataUnpickler(io.BytesIO((root/'datasets/dmd2/captions_coco14_test.pkl').read_bytes())).load()
    eval_set=set(eval_prompts);assert not (set(prompts)|set(pcm_prompts))&eval_set,'train/eval prompt overlap'
    print('TRAIN_SUBSET',len(prompts),'latent_std',latents.float().std().item(),flush=True)
    # The official shard already includes VAE scaling_factor; never scale a second time.
    data={'latents':latents,'prompts':prompts,'pcm_prompts':pcm_prompts,'resolution':1024,'original_indices':[int(i) for i in indices],'dmd_prompt_indices':[int(i) for i in dmd_indices]}
    pipe=StableDiffusionXLPipeline.from_pretrained(base,unet=None,vae=None,variant='fp16',torch_dtype=torch.bfloat16,local_files_only=True)
    pipe.text_encoder.to('cuda');pipe.text_encoder_2.to('cuda')
    keys=['prompt_embeds','negative_prompt_embeds','pooled_prompt_embeds','negative_pooled_prompt_embeds']
    with torch.inference_mode():
        for prefix,texts in [('',prompts),('pcm_',pcm_prompts)]:
            chunks={k:[] for k in keys}
            for i in range(0,len(texts),8):
                encoded=pipe.encode_prompt(texts[i:i+8],device='cuda',do_classifier_free_guidance=True)
                for k,v in zip(keys,encoded):chunks[k].append(v.cpu())
                print('ENCODED',prefix,min(i+8,len(texts)),flush=True)
            data.update({prefix+k:torch.cat(v) for k,v in chunks.items()})
    data['time_ids']=torch.tensor([[1024,1024,0,0,1024,1024]],dtype=torch.float32).repeat(len(prompts),1)
    torch.save(data,root/'datasets/prepared.pt')
    (root/'datasets/prepared_manifest.json').write_text(json.dumps({'source':'DMD2 official real LAION-Aesthetics SDXL latent shard 000 for PCM; separate LAION-Aesthetics6.25 prompt pool for DMD/fake','source_revision':'be22767697a1f3ca656b73c776e15fa335c86c6c','original_shard_samples':len(original['latents']),'samples':len(prompts),'indices':data['original_indices'],'dmd_prompt_indices':data['dmd_prompt_indices'],'resolution':1024,'latent_shape':list(latents.shape),'latent_scaling':'already scaled by original official encoder; unchanged','train_eval_prompt_overlap':0,'prompt_sha256':hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),'conditioning':{k:list(data[k].shape) for k in keys}},indent=2))
    print('SDXL_DATA_PREPARED',flush=True)

if __name__=='__main__':main()
