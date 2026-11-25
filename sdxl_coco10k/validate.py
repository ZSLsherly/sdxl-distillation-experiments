"""Check official preprocessing and reconstructed checkpoint inference before 10K."""
import gc
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import evaluate as e

torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
manifest=json.loads((e.OUT/"manifest.json").read_text())
# Execute the official transform unchanged, capturing the callable passed to FID.
captured={}
def capture(*args,**kwargs):
    captured["transform"]=kwargs["custom_image_tranform"]
    return 0.
package=types.ModuleType("main.coco_eval.cleanfid")
package.fid=types.SimpleNamespace(compute_fid=capture)
sys.modules["main.coco_eval.cleanfid"]=package
source=(Path(__file__).parent/"vendor_reference/coco_evaluator.py").read_text()
ns={};exec(compile(source,"official_coco_evaluator.py","exec"),ns)
resize_ns={}
exec((Path(__file__).parent/"vendor_reference/resize.py").read_text(),resize_ns)
official_resize=resize_ns["build_resizer"]("clean")
rng=np.random.default_rng(21)
for shape in [(1024,1024),(480,640),(200,270),(299,301)]:
    arr=rng.integers(0,256,(*shape,3),dtype=np.uint8)
    for patch in (False,True):
        ns["compute_fid"](None,"unused","cpu",resize_size=512,patch_fid=patch)
        expected=official_resize(captured["transform"](arr))
        actual=e.transformed(Image.fromarray(arr),patch).numpy().transpose(1,2,0)
        assert np.array_equal(expected.astype(np.float32),actual),(shape,patch)
print("OFFICIAL_PREPROCESSING_EXACT_MATCH",flush=True)
f=rng.normal(size=(128,16))
v=e.frechet_distance(f.mean(0),np.cov(f,rowvar=False),f.mean(0),np.cov(f,rowvar=False))
assert abs(v)<1e-8
print("FRECHET_IDENTITY",float(v),flush=True)

report={"official_preprocessing_exact_match":True,"frechet_identity":float(v),"models":{}}
inc=e.inception()
for method in e.METHODS:
    p=e.load_pipeline(method,manifest)
    torch.cuda.reset_peak_memory_stats()
    # Reconstruct a previously saved G160 holdout sample, with the old RNG seed.
    if method in e.RUNS:
        run=e.ROOT/"outputs"/e.RUNS[method]
        dirs=sorted((run/"eval").glob("*g000160*"))
        assert dirs
        rows=[]
        for f in sorted(dirs[-1].glob("rank*.json")):
            rows.extend(json.loads(f.read_text())["samples"])
        row=next(x for x in rows if x["split"]=="holdout")
        with torch.inference_mode():
            images,_=e.generate(p,method,[row["prompt"]],[row["seed"]])
        old=np.asarray(Image.open(dirs[-1]/row["image"]).convert("RGB"),dtype=float)
        new=np.asarray(images[0],dtype=float)
        mae=float(np.abs(old-new).mean())
        psnr=float(10*np.log10(255**2/max(np.mean((old-new)**2),1e-12)))
        # BF16 snapshot vs FP32 master weights and FSDP batching need not be bitwise.
        if psnr<27 or mae>5:
            raise ValueError(f"{method}: checkpoint reconstruction mismatch MAE={mae} PSNR={psnr}")
        check={"prior_G160_MAE":mae,"prior_G160_PSNR":psnr,"prompt":row["prompt"]}
    else:
        check={}
    rows=manifest["samples"][100:108]
    torch.cuda.synchronize();begin=time.monotonic()
    batch=4 if method=="sdxl" else 8
    images=[]
    with torch.inference_mode():
        for i in range(0,len(rows),batch):
            ims,diag=e.generate(p,method,[r["prompt"] for r in rows[i:i+batch]],[r["seed"] for r in rows[i:i+batch]])
            images.extend(ims)
        torch.cuda.synchronize()
        seconds=time.monotonic()-begin
        full,patch=e.features(inc,images)
    dest=e.OUT/"probe_images"/method;dest.mkdir(parents=True,exist_ok=True)
    for row,im in zip(rows,images):im.save(dest/f'{row["index"]:05d}.png')
    check.update(seconds_for_8_images=seconds,images_per_second=8/seconds,
                 batch=batch,peak_GiB=torch.cuda.max_memory_allocated()/2**30,
                 latent_diagnostics=diag,finite_features=bool(np.isfinite(full).all() and np.isfinite(patch).all()))
    report["models"][method]=check
    e.write_json(e.OUT/"validation.json",report)
    print("MODEL_CHECK",method,json.dumps(check),flush=True)
    del p;gc.collect();torch.cuda.empty_cache()
report["status"]="passed"
e.write_json(e.OUT/"validation.json",report)
print("VALIDATION_PASSED",flush=True)
