"""Fresh 10,000-image LAION training split + 32 independent held-out images."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline

root = Path('/data5/shenglin.zeng/sdxl_crossdistill')
sys.path.insert(0, str(root / 'src'))
from sdxl_prepare import load_numpy_pt

out = root / 'datasets/parallel_10k_seed20261018'
out.mkdir(exist_ok=True)
if (out / 'manifest.json').exists():
    manifest = json.loads((out / 'manifest.json').read_text())
    if manifest['train_count'] != 10000:
        raise RuntimeError('Existing dataset has an unexpected training count')
    print('DATA_10K_ALREADY_READY', flush=True)
    raise SystemExit(0)
torch.set_num_threads(4)
paths = [root / f'datasets/dmd2/vae_latents_{i:03d}.pt' for i in (0, 1)]
shards = [load_numpy_pt(p) for p in paths]
offsets = np.cumsum([0] + [len(s['latents']) for s in shards])
locations = [(shard, row) for shard, s in enumerate(shards) for row in range(len(s['latents']))]
seen_prompts, seen_latents, selected = set(), set(), []
for global_idx in np.random.default_rng(20261018).permutation(len(locations)):
    shard, row = locations[global_idx]
    prompt = str(shards[shard]['prompts'][row])
    canonical = ' '.join(prompt.lower().split())
    latent_hash = hashlib.sha256(shards[shard]['latents'][row].tobytes()).hexdigest()
    if canonical in seen_prompts or latent_hash in seen_latents:
        continue
    seen_prompts.add(canonical)
    seen_latents.add(latent_hash)
    selected.append(int(global_idx))
    if len(selected) == 10032:
        break
if len(selected) != 10032:
    raise RuntimeError(f'Need 10032 distinct examples, found {len(selected)}')
holdout, train = selected[:32], selected[32:]
order = train + holdout
prompts = [str(shards[locations[i][0]]['prompts'][locations[i][1]]) for i in order]
latents = np.stack([shards[locations[i][0]]['latents'][locations[i][1]] for i in order])
source_keys = [f'{paths[locations[i][0]].name}:{locations[i][1]}' for i in order]
del shards
result = {'latents': torch.from_numpy(latents), 'prompts': prompts, 'original_indices': order,
          'source_keys': source_keys, 'train_count': 10000,
          'time_ids': torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], dtype=torch.float32)}
pipe = StableDiffusionXLPipeline.from_pretrained(root / 'models/sdxl-base-1.0', unet=None, vae=None,
                                               variant='fp16', torch_dtype=torch.bfloat16, local_files_only=True)
pipe.text_encoder.to('cuda')
pipe.text_encoder_2.to('cuda')
positive, pooled = [], []
with torch.inference_mode():
    for start in range(0, len(order), 32):
        pe, ne, pp, npool = pipe.encode_prompt(prompts[start:start+32], device='cuda', do_classifier_free_guidance=True)
        positive.append(pe.cpu())
        pooled.append(pp.cpu())
        if start == 0:
            result.update(negative_prompt_embeds=ne[:1].cpu(), negative_pooled_prompt_embeds=npool[:1].cpu())
        if start % 512 == 0:
            print('ENCODE_10K', start, len(order), flush=True)
result.update(prompt_embeds=torch.cat(positive), pooled_prompt_embeds=torch.cat(pooled))
torch.save(result, out / 'prepared.pt.partial')
(out / 'prepared.pt.partial').replace(out / 'prepared.pt')
manifest = {'source': 'official DMD2 LAION VAE latent shards 000 and 001',
            'revision': 'be22767697a1f3ca656b73c776e15fa335c86c6c', 'seed': 20261018,
            'source_samples': len(locations), 'train_count': 10000, 'holdout_count': 32,
            'train_original_indices': train, 'holdout_original_indices': holdout,
            'eval_original_indices': holdout[:4], 'source_shard_offsets': offsets.tolist(),
            'dedup': 'exact canonical captions + SHA256 latent bytes over the entire selected split',
            'train_eval_exact_prompt_and_latent_overlap': 0,
            'latent_scaling': 'official stored latents are already VAE-scaled; unchanged',
            'source_files': [str(p) for p in paths]}
(out / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('DATA_10K_READY', str(out), 'train=10000 holdout=32', flush=True)
