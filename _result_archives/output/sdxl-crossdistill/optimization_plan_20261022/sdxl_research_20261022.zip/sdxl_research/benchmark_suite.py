"""Matched execution ablations. Dry run by default; --execute requires idle GPUs."""
import argparse,json,subprocess,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True);ap.add_argument('--gpus',required=True)
    ap.add_argument('--execute',action='store_true')
    ap.add_argument('--candidates',default='legacy,bucket_cycle,block,foreach,teacher_cfg,compile_teacher')
    args=ap.parse_args();base=Path(__file__).resolve().parent
    stamp=datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%dT%H%M%S')
    for name in args.candidates.split(','):
        config=base/'configs'/'performance'/f'{name}.json'
        if not config.exists():ap.error(f'Unknown candidate {name}')
        cmd=[sys.executable,str(base/'launch.py'),'--config',str(config),'--root',args.root,
             '--gpus',args.gpus,'--run',f'perf/{stamp}_{name}']
        if args.execute:cmd.append('--execute')
        subprocess.run(cmd,check=True)

if __name__=='__main__':main()
