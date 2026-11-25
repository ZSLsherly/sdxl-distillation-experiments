"""Six-model COCO10K evaluation. Stream exact clean-FID features; retain fixed PNGs."""
import argparse
import csv
import gc
import hashlib
import io
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionXLPipeline, DDIMScheduler, AutoencoderKL
from safetensors.torch import load_file
from torchvision.transforms.functional import center_crop
from cleanfid.resize import build_resizer
from cleanfid.inception_torchscript import InceptionV3W
from scipy import linalg

ROOT = Path('.')
OUT = Path('.')
BASE = Path('.')
from objectives import VPSchedule
RUNS = {}
METHODS = []
SEED = 20261020
RESIZE = build_resizer("clean")


def feature_blocks(n, rank, world):
    """Globally stable chunk IDs, distributed by block rather than image range."""
    return [(left, min(left + 64, n)) for left in range(0, n, 64)][rank::world]


def frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # clean-FID formula; current SciPy removed sqrtm(disp=False).
    diff = mu1 - mu2
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            raise ValueError("Non-negligible imaginary covariance square root")
        covmean = covmean.real
    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2*np.trace(covmean)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.partial")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def transformed(im, patch):
    # Identical to DMD2 coco_evaluator.py custom transform + clean-FID resizer.
    if patch:
        if min(im.size) >= 299:
            im = center_crop(im, 299)
    else:
        im = center_crop(im, min(im.size))
        im = im.resize((512, 512), Image.Resampling.LANCZOS)
    arr = RESIZE(np.asarray(im))
    return torch.from_numpy(arr.transpose(2, 0, 1).copy()).float()


def inception():
    cache = ROOT / "models/metrics/clean_fid"
    cache.mkdir(parents=True, exist_ok=True)
    return InceptionV3W(str(cache), download=True, resize_inside=False).eval().cuda()


@torch.inference_mode()
def features(model, images):
    values = []
    for patch in (False, True):
        x = torch.stack([transformed(im, patch) for im in images]).cuda()
        z = model(x).float().cpu().numpy()
        if z.shape != (len(images), 2048) or not np.isfinite(z).all():
            raise ValueError("Invalid Inception features")
        values.append(z)
    return values


def load_student(pipe, method, manifest):
    rec = manifest["models"][method]
    path = Path(rec["checkpoint"])
    shapes = json.loads((path / "shapes_rank0.json").read_text())
    shards = [torch.load(path / f"student_rank{i}.pt", map_location="cpu",
                         weights_only=True, mmap=True) for i in range(rec["world_size"])]
    state = {}
    for k, meta in shapes.items():
        if meta["placements"] and meta["placements"][0].startswith("S("):
            dim = int(meta["placements"][0][2:-1])
            value = torch.cat([s[k] for s in shards], dim=dim)
        else:
            value = shards[0][k]
            assert all(torch.equal(value, s[k]) for s in shards[1:]), k
        assert list(value.shape) == meta["shape"], k
        state[k] = value
    pipe.unet.load_state_dict(state, strict=True)
    del state, shards
    gc.collect()


def load_pipeline(method, manifest):
    # Load directly in FP32: BF16 -> FP32 would already have rounded VAE weights.
    vae = AutoencoderKL.from_pretrained(BASE, subfolder="vae", variant="fp16",
                                       torch_dtype=torch.float32, local_files_only=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        BASE, variant="fp16", torch_dtype=torch.bfloat16,
        local_files_only=True, add_watermarker=False, vae=vae)
    if method in RUNS:
        load_student(pipe, method, manifest)
        pipe._research_knots = manifest['models'][method]['config']['schedule']
    elif method == "dmd2":
        state = load_file(str(ROOT / "models/baselines/dmd2/unet.safetensors"))
        pipe.unet.load_state_dict(state, strict=True)
        del state
    pipe.to("cuda")
    pipe.vae.to(torch.float32)
    pipe.set_progress_bar_config(disable=True)
    pipe.unet.eval().requires_grad_(False)
    return pipe


@torch.inference_mode()
def generate(pipe, method, prompts, seeds):
    # Per-prompt RNG makes sample identity independent of batch size/GPU sharding.
    noise = torch.stack([torch.randn((4, 128, 128), device="cuda",
                         generator=torch.Generator(device="cuda").manual_seed(s))
                         for s in seeds])
    if method == "sdxl":
        x = pipe(prompt=prompts, height=1024, width=1024,
                 latents=noise.to(torch.bfloat16), num_inference_steps=50,
                 guidance_scale=6.0, output_type="latent").images.float()
    else:
        cond, _, pooled, _ = pipe.encode_prompt(prompts, device="cuda",
                                                do_classifier_free_guidance=False)
        time_ids = torch.tensor([[1024,1024,0,0,1024,1024]], device="cuda",
                                dtype=torch.float32).repeat(len(prompts),1)
        scheduler = DDIMScheduler.from_pretrained(BASE, subfolder="scheduler", local_files_only=True)
        vp = VPSchedule(scheduler.alphas_cumprod)
        x = noise
        knots = pipe._research_knots if method in RUNS else [999,749,499,249,-1]
        for j, t in enumerate(knots[:-1]):
            tt = torch.full((len(prompts),), t, device="cuda", dtype=torch.long)
            with torch.autocast("cuda",dtype=torch.bfloat16):
                eps = pipe.unet(x, tt, encoder_hidden_states=cond,
                    added_cond_kwargs={"text_embeds":pooled,"time_ids":time_ids},
                    return_dict=False)[0].float()
            if method in RUNS:
                x = vp.jump(x, eps, tt, knots[j+1])
            else:
                # Official DMD2 research sampler: predict x0 then independent re-noise.
                x0 = vp.clean(x, eps, tt)
                if j < 3:
                    z = torch.stack([torch.randn((4,128,128),device="cuda",
                        generator=torch.Generator(device="cuda").manual_seed(s+100000*(j+1)))
                        for s in seeds])
                    x = vp.noise(x0,z,knots[j+1])
                else:
                    x = x0
    if not torch.isfinite(x).all():
        raise FloatingPointError(f"{method}: nonfinite latent")
    images=[]
    for i in range(len(x)):
        rgb=pipe.vae.decode(x[i:i+1].float()/pipe.vae.config.scaling_factor,return_dict=False)[0]
        if not torch.isfinite(rgb).all():
            raise FloatingPointError(f"{method}: nonfinite VAE")
        # Same uint8 conversion for every method and official DMD2.
        arr=((rgb[0]+1)*127.5).clamp(0,255).to(torch.uint8).permute(1,2,0).cpu().numpy()
        images.append(Image.fromarray(arr))
    return images, {"latent_std":float(x.std()),"latent_abs_max":float(x.abs().max())}


@torch.inference_mode()
def worker(args):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    manifest=json.loads((OUT/"manifest.json").read_text())
    inc=inception()
    visual=set(manifest["visual_indices"])
    # Fixed global blocks keep cache filenames stable when worker count changes.
    blocks=feature_blocks(manifest["n"], args.rank, args.world)
    if args.limit:blocks=blocks[:max(1,(args.limit+63)//64)]
    target=sum(right-left for left,right in blocks)
    for method in args.methods.split(","):
        dest=OUT/("probe" if args.probe else "features")/method
        dest.mkdir(parents=True,exist_ok=True)
        valid={f"{l:05d}_{r:05d}.npz" for l,r in feature_blocks(manifest["n"],0,1)}
        if any(p.name not in valid for p in dest.glob("*.npz")):
            raise ValueError("Legacy rank-dependent feature blocks: use a fresh evaluation output directory")
        pipe=None if method=="real" else load_pipeline(method,manifest)
        batch=args.teacher_batch if method=="sdxl" else args.batch
        torch.cuda.reset_peak_memory_stats()
        begin=time.monotonic();done=0
        for left,right in blocks:
            fpath=dest/f"{left:05d}_{right:05d}.npz"
            if fpath.exists():
                with np.load(fpath) as old:
                    assert np.array_equal(old["indices"],np.arange(left,right))
                done+=right-left;continue
            full=[];patch=[];diag=[]
            at=left
            while at<right:
                ids=list(range(at,min(at+batch,right)))
                rows=[manifest["samples"][i] for i in ids]
                try:
                    if method=="real":
                        images=[Image.open(OUT/"real"/r["real_file"]).convert("RGB") for r in rows]
                        info={}
                    else:
                        images,info=generate(pipe,method,[r["prompt"] for r in rows],[r["seed"] for r in rows])
                    a,b=features(inc,images)
                except torch.cuda.OutOfMemoryError:
                    if batch==1:raise
                    batch=max(1,batch//2);gc.collect();torch.cuda.empty_cache()
                    print("OOM_REDUCE",method,args.rank,batch,flush=True)
                    continue
                full.append(a);patch.append(b);diag.append(info)
                for idx,im in zip(ids,images):
                    if idx in visual or args.probe:
                        pd=OUT/("probe_images" if args.probe else "images")/method
                        pd.mkdir(parents=True,exist_ok=True)
                        im.save(pd/f"{idx:05d}.png",compress_level=3)
                at+=len(ids)
            with fpath.with_suffix(".partial").open("wb") as f:
                np.savez(f,indices=np.arange(left,right),full=np.concatenate(full),patch=np.concatenate(patch))
            fpath.with_suffix(".partial").replace(fpath)
            done+=right-left
            elapsed=time.monotonic()-begin
            progress={"method":method,"rank":args.rank,"done":done,"target":target,
                      "last_index":right-1,"seconds":elapsed,"images_per_second":done/elapsed,
                      "batch":batch,"peak_GiB":torch.cuda.max_memory_allocated()/2**30,
                      "last_latent_diagnostics":diag[-1],"updated_unix":time.time()}
            write_json(OUT/f"status_{args.rank}.json",progress)
            print("PROGRESS",json.dumps(progress),flush=True)
        del pipe;gc.collect();torch.cuda.empty_cache()
        print("METHOD_DONE",method,args.rank,done,flush=True)
    write_json(OUT/f"worker_{args.rank}_done.json",{"time":time.time(),"probe":args.probe,"methods":args.methods})


def metrics():
    torch.set_num_threads(8)
    expected_n = json.loads((OUT/'manifest.json').read_text())['n']
    summary={}
    real={}
    for method in ["real"]+METHODS:
        chunks=[]
        for p in sorted((OUT/"features"/method).glob("*.npz")):
            with np.load(p) as z:chunks.append({k:z[k].copy() for k in z.files})
        if not chunks:raise ValueError(f"No features: {method}")
        ids=np.concatenate([x["indices"] for x in chunks])
        assert np.array_equal(ids,np.arange(expected_n)),(method,"incomplete or duplicate features",len(ids))
        results={"n":len(ids)}
        for kind in ["full","patch"]:
            fs=np.concatenate([x[kind] for x in chunks]).astype(np.float64)
            assert fs.shape==(expected_n,2048) and np.isfinite(fs).all()
            mu,cov=fs.mean(0),np.cov(fs,rowvar=False)
            if method=="real":
                real[kind]=(mu,cov)
                # Exact same-feature sanity check.
                print("REAL_STATS",kind,len(fs),flush=True)
            else:
                val=float(frechet_distance(real[kind][0],real[kind][1],mu,cov))
                if not np.isfinite(val) or val<-.01:raise ValueError(val)
                results["FID" if kind=="full" else "Patch_FID"]=val
            del fs
        if method!="real":
            summary[method]=results
            write_json(OUT/"scores.json",summary)
            print("SCORE",method,json.dumps(results),flush=True)
    write_json(OUT/"complete.json",{"completed_unix":time.time(),"scores":summary,"n_per_model":expected_n})

