"""Summarize measured full-ratio benchmark runs; never mix trace/profile throughput."""
import argparse,json,statistics
from pathlib import Path

def summarize(path):
    cfg=json.loads((path/'config.json').read_text())
    status=json.loads((path/'benchmark.json').read_text())
    if status['status']!='complete' or status['student_updates']!=cfg['benchmark_cycles']:
        raise ValueError(f'{path}: incomplete benchmark; do not report it as a completed timing run')
    if status.get('trace_enabled') or cfg.get('profile'):raise ValueError(f'{path}: instrumentation run, not throughput')
    rows=[json.loads(line) for line in (path/'rank0.jsonl').read_text().splitlines()]
    warmup=status['warmup_cycles']
    gs=[r for r in rows if r['phase']=='student_done' and r['g']>warmup]
    stages=cfg.get('student_stages',[2,3,4])
    if {r['stage'] for r in gs}!=set(stages):
        raise ValueError(f'{path}: measured window must cover configured student stages {stages}')
    fs=[r for r in rows if r['phase'] in ('fake','fake_D') and r['fake_update']>warmup*cfg['fake_updates']]
    return {'run':str(path),'world':cfg['world_size'],'batch':cfg['effective_batch'],
        'fake_G_ratio':cfg['fake_updates'],'schedule':cfg['schedule'],
        'measured_G_updates':len(gs),'median_cycle_s':statistics.median(r['cycle_seconds'] for r in gs),
        'mean_cycle_s':statistics.mean(r['cycle_seconds'] for r in gs),
        'mean_fake_update_s':statistics.mean(r['seconds'] for r in fs),
        'mean_G_s':statistics.mean(r['seconds'] for r in gs),
        'by_stage':{str(s):statistics.median(r['cycle_seconds'] for r in gs if r['stage']==s) for s in stages},
        'allocated_GiB':status['peak_allocated_GiB'],
        'reserved_GiB_all_ranks':status['peak_reserved_GiB'],
        'reserved_GiB_rank0':max(r['peak_reserved_GiB'] for r in gs),
        'warmup_s':status['warmup_seconds'], 'includes_eval_and_checkpoint':False}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('runs',nargs='+',type=Path);ap.add_argument('--out',type=Path)
    args=ap.parse_args();data=[summarize(p) for p in args.runs]
    value=json.dumps(data,ensure_ascii=False,indent=2)
    if args.out:args.out.write_text(value,encoding='utf-8')
    print(value)
