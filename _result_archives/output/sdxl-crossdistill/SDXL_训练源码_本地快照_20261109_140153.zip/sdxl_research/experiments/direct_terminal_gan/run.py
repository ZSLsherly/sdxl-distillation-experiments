"""H20 direct-terminal ablation. Default is a dry run, --short tests three full updates."""
import argparse
import hashlib
import json
import os
import shlex
from shutil import copyfile
import subprocess
import sys
import time
from pathlib import Path

here = Path(__file__).resolve().parent
base = Path('/root/autodl-tmp/h20_C_mix1_formal_20261103')
root = Path(os.environ['SDXL_ROOT'])
sys.path.insert(0, str(base))
from launch import flags, gpu_check

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--short', action='store_true')
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    baseline = root/'outputs/h20_C_mix1_terminal_GAN_G160'
    actual = json.loads((baseline/'config.json').read_text())
    # The launch template gives CLI field names. Take values from the real run.
    fields = json.loads((base/'config.json').read_text())
    cfg = {key:actual[key] for key in fields}
    if cfg['schedule'] != [999,749,499,249,-1] or not cfg['flash_terminal_gan']:
        raise ValueError('Unexpected baseline schedule or GAN route')
    cfg['schedule'] = cfg['schedule'][:-1]
    cfg.update(flash_terminal_gan=False, direct_terminal_gan=True, benchmark=args.short,
               turning_profile=str(baseline/'source/turning_distribution.json'),
               max_student_updates=3 if args.short else 160)
    name = 'h20_C_mix1_direct_terminal_GAN_' + ('probe3' if args.short else 'G160')
    out = root/'outputs'/name
    # Sibling files are also archived by train.py in its source snapshot.
    src = out.parent/(name+'_code')
    cmd = [sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
           str(src/'train.py'),'--root',str(root),'--run',name,*flags(cfg)]
    print(shlex.join(cmd), flush=True)
    if not args.execute:return
    from turning_batch64_workflow import descendants, foreign_on, stop_own
    gpu_uuid=gpu_check([0],90)[0][1]
    if out.exists():raise FileExistsError(out)
    src.mkdir(parents=True,exist_ok=False)
    hashes=json.loads((baseline/'source_sha256.json').read_text())
    for name_in_snapshot,digest in hashes.items():
        original=baseline/'source'/name_in_snapshot
        if hashlib.sha256(original.read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Baseline snapshot has changed: {original}')
        if original.suffix=='.py' and original.name!='train.py':
            copyfile(original,src/original.name)
    for name_in_experiment in ('train.py','terminal_gan.py'):
        copyfile(here/name_in_experiment,src/name_in_experiment)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
    lib='/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib'
    env['LD_PRELOAD']=f'{lib}/libcublasLt.so.12:{lib}/libcublas.so.12'
    process=subprocess.Popen(cmd,env=env,start_new_session=True)
    try:
        while process.poll() is None:
            if foreign_on({gpu_uuid},descendants(process.pid)):
                raise RuntimeError('Another GPU job appeared during the run')
            time.sleep(4)
        if process.returncode:
            raise subprocess.CalledProcessError(process.returncode,cmd)
    except BaseException:
        stop_own(process)
        raise
    status=json.loads((out/'status.json').read_text())
    g=cfg['benchmark_cycles'] if args.short else cfg['max_student_updates']
    if (status['status'],status['student_updates'],status['fake_updates'],status['D_updates'])!=('complete',g,2*g,g):
        raise RuntimeError(f'Run stopped short of {g} full updates: {out}')

if __name__=='__main__':main()
