"""Two independent inference workers; stop our workers on GPU conflicts."""
import os,sys,time,signal,subprocess,json,csv,io
from pathlib import Path
ROOT=Path('/data1/shenglin.zeng/teacher_turning_20261026');GPUS=[0,1]
def apps():
    out=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
    return [(r[0].strip(),int(r[1])) for r in csv.reader(io.StringIO(out)) if len(r)>=2]
def tree(pid):
    owned={pid};todo=[pid]
    while todo:
        p=todo.pop()
        try:children=(Path('/proc')/str(p)/'task'/str(p)/'children').read_text().split()
        except OSError:continue
        for v in children:
            v=int(v)
            if v not in owned:owned.add(v);todo.append(v)
    return owned
rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
cards={int(r[0]):[v.strip() for v in r] for r in csv.reader(io.StringIO(rows))}
uuids={cards[i][1] for i in GPUS}
if any(uuid in uuids for uuid,pid in apps()):raise RuntimeError('Selected GPU occupied')
if any(int(cards[i][2])<22000 or int(cards[i][3])>5 for i in GPUS):raise RuntimeError('GPU headroom insufficient')
OUT=ROOT/'results';OUT.mkdir(exist_ok=True);children=[];logs=[]
try:
    for rank,gpu in enumerate(GPUS):
        log=(ROOT/f'worker{rank}.log').open('a');logs.append(log)
        env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2',
             'MKL_NUM_THREADS':'2','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
             'TMPDIR':str(ROOT/'tmp'),'HF_HOME':str(ROOT/'cache')}
        cmd=[sys.executable,'-u',str(ROOT/'sdxl_turning_resample.py'),'--root',str(ROOT),'--out',str(OUT),
             '--worker',str(rank),'--workers','2']
        children.append(subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True))
    (ROOT/'pids.json').write_text(json.dumps([p.pid for p in children]))
    while any(p.poll() is None for p in children):
        active=apps();owned=set().union(*(tree(p.pid) for p in children))
        foreign=[pid for uuid,pid in active if uuid in uuids and pid not in owned]
        if foreign:raise RuntimeError('Other workload entered selected GPU: '+str(foreign))
        if any(p.poll() not in (None,0) for p in children):raise RuntimeError('Sampling worker failed')
        if (ROOT/'STOP_REQUEST').exists():raise RuntimeError('Stop requested')
        time.sleep(10)
    if any(p.returncode for p in children):raise RuntimeError('Sampling worker failed')
    (ROOT/'complete.json').write_text(json.dumps({'status':'complete','time':time.time()}))
except BaseException as e:
    for p in children:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGTERM)
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    (ROOT/'error.json').write_text(json.dumps({'error':str(e),'time':time.time()}))
    raise
finally:
    for f in logs:f.close()
