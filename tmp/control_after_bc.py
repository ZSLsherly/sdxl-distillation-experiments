"""Evaluate the matched no-prior control on the exact B/C protocol."""
import json
import math
import subprocess
import sys
from pathlib import Path
from run_bc_coco_then_winner import (code, env, evalout, inputs, job, on_idle_gpu,
                                     previous, save, sha, verify)

def prepare_control():
    manifest = json.loads((evalout / 'manifest.json').read_text())
    assert manifest['n'] == len(manifest['samples']) == 10000 and len(manifest['visual_indices']) == 128
    run = previous / 'uniform_control'
    dest = Path(verify(run))
    cfg = json.loads((run / 'config.json').read_text())
    assert cfg['turning_profile'] is None and cfg['schedule'] == [999,749,499,249,-1]
    manifest['models'] = {'A': dict(checkpoint=str(dest), world_size=1, student_updates=160,
                                    config=cfg, shards=[dict(name='student_rank0.pt',sha256=sha(dest/'student_rank0.pt'))])}
    out = job / 'coco10k_A'
    (out/'features').mkdir(parents=True,exist_ok=True)
    real = out/'features/real'
    reference = (evalout/'features/real').resolve(strict=True)
    if not real.exists(): real.symlink_to(reference,target_is_directory=True)
    assert real.resolve(strict=True) == reference
    path = out/'manifest.json'
    if path.exists(): assert json.loads(path.read_text()) == manifest
    else: save(path,manifest)
    return out

def run_control():
    out = prepare_control()
    if not (out/'complete.json').exists():
        # The original parent holds the queue lock throughout both evaluations.
        opts = ['--root',str(inputs),'--out',str(out)]
        base = [sys.executable,'-u',str(code/'evaluate.py')]
        on_idle_gpu(base+['worker',*opts,'--methods','A','--batch','4'],out/'worker.log')
        with (out/'metrics.log').open('a') as log:
            subprocess.run(base+['metrics',*opts],cwd=code,env={**env,'CUDA_VISIBLE_DEVICES':''},
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    record=json.loads((out/'complete.json').read_text())['scores']['A']
    assert record['n']==10000 and all(math.isfinite(record[k]) for k in ('FID','Patch_FID'))
    return record
