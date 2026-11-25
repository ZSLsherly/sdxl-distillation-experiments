"""One explicit run; defaults to printing the command. No background queue."""
import argparse, csv, io, json, os, shlex, subprocess, sys, time
from datetime import datetime, timezone, timedelta
from pathlib import Path

BASE = Path(__file__).resolve().parent


def flags(config):
    result=[]
    for key,value in config.items():
        name='--'+key.replace('_','-')
        if value is None:continue
        if isinstance(value,bool):
            if value:result.append(name)
            elif key=='fake_offload':result.append('--no-fake-offload')
        elif isinstance(value,list):result.extend([name,*map(str,value)])
        else:result.extend([name,str(value)])
    return result


def gpu_check(gpus, minimum_gib):
    selection='--id='+','.join(map(str,gpus))
    text=subprocess.check_output(['nvidia-smi',selection,'--query-gpu=index,uuid,memory.free,utilization.gpu,pstate',
                                  '--format=csv,noheader,nounits'],text=True)
    rows={int(r[0]):[x.strip() for x in r] for r in csv.reader(io.StringIO(text))}
    apps=subprocess.check_output(['nvidia-smi',selection,'--query-compute-apps=gpu_uuid,pid',
                                  '--format=csv,noheader,nounits'],text=True)
    busy={r[0].strip() for r in csv.reader(io.StringIO(apps)) if len(r)>=2}
    selected=[]
    for index in gpus:
        row=rows[index]
        if not row[2].isdigit() or not row[3].isdigit() or 'reset' in row[4].lower():
            raise RuntimeError(f'GPU {index}: unhealthy status {row[2:]}')
        if row[1] in busy or int(row[3])>5:
            raise RuntimeError(f'GPU {index}: another compute task is present; not launching')
        if int(row[2])/1024 < minimum_gib:raise RuntimeError(f'GPU {index}: insufficient headroom')
        selected.append(row)
    return selected


def command(config, root, run, world):
    if config['batch'] % (world*config['micro_batch']):raise ValueError('Invalid global batch/microbatch/world')
    return [sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={world}',
            str(BASE/'train.py'),'--root',str(root),'--run',run,*flags(config)]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True);ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--gpus',required=True);ap.add_argument('--run');ap.add_argument('--execute',action='store_true')
    ap.add_argument('--short',action='store_true',help='4 real updates, including R1/GAN/eval/final state; fresh isolated run')
    ap.add_argument('--min-free-gib',type=float,default=36)
    args=ap.parse_args();gpus=[int(x) for x in args.gpus.split(',')]
    if len(set(gpus))!=len(gpus) or not gpus:ap.error('GPU indices must be unique')
    config=json.loads(args.config.read_text(encoding='utf-8'))
    if args.short:
        # A performance config must not silently keep benchmark=True here:
        # --short promises evaluation and checkpoint IO as well as four updates.
        config.update(benchmark=False,max_student_updates=4,save_every_updates=4)
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    run=args.run or f'research/{args.config.stem}_{"short" if args.short else "train"}_{stamp}'
    root=args.root.resolve()
    out=root/'outputs'/run
    if not out.resolve().is_relative_to((root/'outputs').resolve()):raise ValueError('Run must stay under outputs')
    cmd=command(config,root,run,len(gpus))
    print(shlex.join(cmd),flush=True)
    if not args.execute:return
    if out.exists():raise FileExistsError(out)
    check=gpu_check(gpus,args.min_free_gib)
    env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=args.gpus,NCCL_P2P_DISABLE='1',NCCL_IB_DISABLE='1',OMP_NUM_THREADS='4')
    env['TORCHDYNAMO_DISABLE']='0' if config.get('compile_teacher') else '1'
    # Keep compiler/temp/cache writes in the selected project, not the system root.
    for variable,folder in [('TMPDIR','tmp'),('TORCHINDUCTOR_CACHE_DIR','cache/inductor'),('TRITON_CACHE_DIR','cache/triton')]:
        path=root/folder;path.mkdir(parents=True,exist_ok=True);env[variable]=str(path)
    logs=root/'logs';logs.mkdir(parents=True,exist_ok=True)
    name=run.replace('/','_')
    (logs/f'{name}.launch.json').write_text(json.dumps({'command':cmd,'gpus':check,'config':config},indent=2))
    begin=time.monotonic()
    with (logs/f'{name}.log').open('w') as log:
        rc=subprocess.call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    (logs/f'{name}.result.json').write_text(json.dumps({
        'returncode':rc,'wall_seconds_including_initialization':time.monotonic()-begin},indent=2))
    print('exit',rc,'log',logs/f'{name}.log')
    raise SystemExit(rc)

if __name__=='__main__':main()
