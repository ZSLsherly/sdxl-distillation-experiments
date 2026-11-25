"""Two authorized short tests on idle GPUs; no formal training or CPU offload."""
import csv, io, json, os, signal, subprocess, sys, time
from pathlib import Path

ROOT = Path('/data3/shenglin.zeng/sdxl_turning_ablation_20261028')
NAME = 'smoke_four_gpu_expandable_retry1_20261029'
OUT = ROOT / 'outputs' / NAME
CODE = ROOT / 'code/sdxl_research'
sys.path.insert(0, str(CODE))
from launch import gpu_check, command

ENV = dict(os.environ, CUDA_VISIBLE_DEVICES='1,2,4,5', NCCL_P2P_DISABLE='1', NCCL_IB_DISABLE='1',
           OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
           TORCHDYNAMO_DISABLE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
           HF_HOME=str(ROOT/'cache'), TMPDIR=str(ROOT/'tmp'), TOKENIZERS_PARALLELISM='false')
ENV.pop('SDXL_SHARED_YIELD_FILE', None)

def event(kind, **data):
    row=dict(time=time.time(), event=kind, **data)
    with (OUT/'workflow.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
    print(json.dumps(row), flush=True)

def descendants(pid):
    owned={pid}; todo=[pid]
    while todo:
        p=todo.pop()
        try: children=(Path('/proc')/str(p)/'task'/str(p)/'children').read_text().split()
        except OSError: continue
        for child in map(int,children):
            if child not in owned: owned.add(child); todo.append(child)
    return owned

def run(arm):
    cards=gpu_check([1,2,4,5],22); uuids={r[1] for r in cards}
    config=json.loads((CODE/f'configs/turning_{arm}_2gpu_no_offload_short.json').read_text())
    assert all(config[k] is False for k in ('fake_offload','offload_student','offload_student_optimizer'))
    config.update(benchmark_batch=8, benchmark_cycles=2, benchmark_warmup=1, max_student_updates=2)
    config['turning_profile']=str(CODE/f'turning_profiles/{arm}_512.json')
    run_name=f'{NAME}/{arm}'
    result_dir=ROOT/'outputs'/run_name
    if result_dir.exists(): raise FileExistsError(result_dir)
    cmd=command(config,ROOT,run_name,4)
    with (OUT/f'{arm}.log').open('w') as log:
        p=subprocess.Popen(cmd,env=ENV,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        begin=time.monotonic()
        event('started',arm=arm,pid=p.pid,gpus=cards,config=config)
        try:
            while p.poll() is None:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
                owned=descendants(p.pid)
                foreign=[int(r[1]) for r in csv.reader(io.StringIO(apps)) if len(r)>1 and r[0].strip() in uuids and int(r[1]) not in owned and Path('/proc',r[1].strip()).exists()]
                available=next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                if foreign or available<24 or (OUT/'USER_STOP').exists() or time.monotonic()-begin>2400:
                    raise RuntimeError(f'Own test stopped: foreign={foreign}, RAM_available_GiB={available:.1f}, elapsed={time.monotonic()-begin:.0f}')
                time.sleep(5)
            if p.returncode: raise RuntimeError(f'{arm} exit={p.returncode}; see arm log')
        except BaseException:
            if p.poll() is None:
                for pid in sorted(descendants(p.pid),reverse=True):
                    try: os.kill(pid,signal.SIGTERM)
                    except ProcessLookupError: pass
                try: os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError: pass
                try: p.wait(timeout=20)
                except subprocess.TimeoutExpired: os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    result=json.loads((result_dir/'benchmark.json').read_text())
    assert result['status']=='complete' and result['student_updates']==2 and result['fake_updates']==4 and result['D_updates']==0, result
    event('complete',arm=arm,result=result)
    return result

try:
    arms=['normalized_u']
    assert all(arm in ('normalized_u','raw_angle') for arm in arms)
    results={arm:run(arm) for arm in arms}
    for arm in ('normalized_u','raw_angle'):
        saved=OUT/arm/'benchmark.json'
        if arm not in results and saved.exists(): results[arm]=json.loads(saved.read_text())
    (OUT/'results.json').write_text(json.dumps(results,indent=2))
except BaseException as e:
    event('failed',reason=str(e))
    raise
