"""Config-driven COCO evaluation using the existing audited FID/Patch-FID path.

Each custom model explicitly names its checkpoint directory. Its sampling
schedule is read from the saved training config, never inferred from its name.
Run prepare once, independent worker processes, then metrics after all finish.
"""
import argparse, csv, hashlib, io, json, time, zipfile
from pathlib import Path
import evaluation_core as core


def prepare(args, specs):
    out=core.OUT
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists():raise FileExistsError('Reuse existing manifest or choose a fresh output')
    zpath=core.ROOT/'datasets/dmd2/coco10k.zip'
    with zipfile.ZipFile(zpath) as z:
        csv_bytes=z.read('coco10k/subset.csv')
        all_rows=list(csv.DictReader(io.StringIO(csv_bytes.decode())))
        if len(all_rows)!=10000 or len({r['file_name'] for r in all_rows})!=10000:raise ValueError('Expected fixed COCO10K')
        # Development subset is deterministic; final 10K keeps the original order.
        ids=list(range(10000)) if args.n==10000 else sorted(core.np.random.default_rng(20261022).choice(10000,args.n,replace=False).tolist())
        rows=[all_rows[i] for i in ids]
        realdir=out/'real';realdir.mkdir(exist_ok=True)
        for r in rows:
            if Path(r['file_name']).name!=r['file_name']:raise ValueError('Invalid image name')
            (realdir/r['file_name']).write_bytes(z.read('coco10k/subset/'+r['file_name']))
    models={}
    for name,spec in specs.items():
        if name=='sdxl':
            models[name]={'weights':str(core.BASE)};continue
        if name=='dmd2':
            p=core.ROOT/'models/baselines/dmd2/unet.safetensors'
            models[name]={'weights':str(p),'sha256':core.sha(p)};continue
        ck=Path(spec['checkpoint'])
        if not ck.is_absolute():ck=core.ROOT/ck
        complete=json.loads((ck/'complete.json').read_text())
        config=json.loads((ck.parent.parent/'config.json').read_text())
        knots=config['schedule']
        if len(knots)!=5 or knots[-1]!=-1:raise ValueError('Expected four-stage checkpoint')
        models[name]={'checkpoint':str(ck),'world_size':complete['world_size'],
            'student_updates':complete['student_updates'],'config':config,
            'shards':[{'name':f'student_rank{i}.pt','sha256':core.sha(ck/f'student_rank{i}.pt')} for i in range(complete['world_size'])]}
    visual=sorted(core.np.random.default_rng(core.SEED).choice(args.n,min(args.n,128),replace=False).tolist())
    manifest={'created_unix':time.time(),'n':args.n,'source_zip_sha256':core.sha(zpath),
        'csv_sha256':hashlib.sha256(csv_bytes).hexdigest(),'models':models,'visual_indices':visual,
        'samples':[{'index':i,'source_index':j,'prompt':r['caption'],'real_file':r['file_name'],'seed':core.SEED+j} for i,(j,r) in enumerate(zip(ids,rows))],
        'protocol':{'FID':'DMD2 center-square -> Lanczos512 -> clean Inception299',
         'Patch_FID':'DMD2 native center299 -> clean Inception299; not a general texture score',
         'custom_sampler':'deterministic VP/DDIM; per-model config.schedule; CFG1',
         'dmd2_sampler':'official research x0/re-noise; [999,749,499,249]; CFG1',
         'sdxl_sampler':'50 steps default scheduler; CFG6','resolution':1024,
         'development_only':args.n!=10000,'seeds':'20261020 + original COCO10K row index'}}
    core.write_json(out/'manifest.json',manifest)
    print('PREPARED',out,args.n)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('command',choices=['prepare','worker','metrics'])
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--models',type=Path,help='JSON mapping method name to {checkpoint: ...}; sdxl/dmd2 use {}')
    ap.add_argument('--n',type=int,choices=[1000,10000],default=10000)
    ap.add_argument('--rank',type=int,default=0);ap.add_argument('--world',type=int,default=1)
    ap.add_argument('--methods',help='Comma-separated subset for this worker; default real + all methods')
    ap.add_argument('--batch',type=int,default=4);ap.add_argument('--teacher-batch',type=int,default=2)
    args=ap.parse_args()
    if args.world<1 or not 0<=args.rank<args.world:ap.error('Invalid rank/world')
    core.ROOT=args.root.resolve();core.OUT=args.out.resolve();core.BASE=core.ROOT/'models/sdxl-base-1.0'
    if args.command=='prepare':
        if not args.models:ap.error('--models is required for prepare')
        specs=json.loads(args.models.read_text())
    else:specs=json.loads((core.OUT/'manifest.json').read_text())['models']
    core.RUNS={name:spec for name,spec in specs.items() if name not in ('sdxl','dmd2')}
    core.METHODS=list(specs)
    args.methods=args.methods or ','.join(['real']+core.METHODS)
    if not set(args.methods.split(',')).issubset({'real',*core.METHODS}):ap.error('Unknown method')
    args.limit=0;args.probe=False
    if args.command=='prepare':prepare(args,specs)
    elif args.command=='worker':core.worker(args)
    else:core.metrics()

if __name__=='__main__':main()
