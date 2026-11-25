"""One authorized short test, then one G160 run on the checked idle GPU."""
import csv,io,json,os,signal,subprocess,sys,time
from pathlib import Path
ROOT=Path('/data5/shenglin.zeng/sdxl_crossdistill')
NAME='boundary_dmd_10k_a100_20261025_v1'
OUT=ROOT/'outputs'/NAME
CODE=OUT/'code/sdxl_research'
sys.path.insert(0,str(CODE))
from launch import gpu_check,command
GPU=2
ENV={**os.environ,'CUDA_VISIBLE_DEVICES':str(GPU),'NCCL_P2P_DISABLE':'1',
     'NCCL_IB_DISABLE':'1','OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2',
     'OPENBLAS_NUM_THREADS':'2','TORCHDYNAMO_DISABLE':'1',
     'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
     'TMPDIR':str(ROOT/'tmp'),'TOKENIZERS_PARALLELISM':'false'}
def write(name,data):
    (OUT/name).write_text(json.dumps(data,indent=2))
def event(kind,**data):
    row=dict(time=time.time(),event=kind,**data)
    with (OUT/'workflow.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)
def run(label,config):
    cards=gpu_check([GPU],36)
    uuid=cards[0][1]
    run_name=f'{NAME}/{label}'
    if (ROOT/'outputs'/run_name).exists():raise FileExistsError(run_name)
    cmd=command(config,ROOT,run_name,1)
    write(label+'_config.json',config)
    with (OUT/f'{label}.log').open('w') as log:
        p=subprocess.Popen(cmd,env=ENV,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        event('started',phase=label,pid=p.pid,gpu=GPU,command=cmd)
        try:
            while p.poll() is None:
                apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,timeout=10)
                foreign=[]
                for row in csv.reader(io.StringIO(apps)):
                    if len(row)<2 or row[0].strip()!=uuid:continue
                    pid=int(row[1])
                    try:owned=os.getpgid(pid)==p.pid
                    except ProcessLookupError:continue
                    if not owned:foreign.append(pid)
                available=next(int(x.split()[1])/1024**2 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:'))
                if foreign or available<32 or (OUT/'USER_STOP').exists():
                    raise RuntimeError(f'Stopping own group: foreign={foreign}, host_available_GiB={available:.1f}, user_stop={(OUT/"USER_STOP").exists()}')
                time.sleep(10)
            if p.returncode:raise RuntimeError(f'{label} exited {p.returncode}; inspect log')
        except BaseException:
            if p.poll() is None:
                os.killpg(p.pid,signal.SIGTERM)
                try:p.wait(timeout=20)
                except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
            raise
    event('finished',phase=label)
    return ROOT/'outputs'/run_name
try:
    formal=json.loads((CODE/'configs/boundary_dmd_10k_g160_1gpu.json').read_text())
    short={**formal,'benchmark':True,'benchmark_batch':8,'benchmark_cycles':4,'benchmark_warmup':1,
           'max_student_updates':4,'hours':1}
    short_out=run('short',short)
    result=json.loads((short_out/'benchmark.json').read_text())
    if result['status']!='complete' or result['student_updates']!=4 or result['fake_updates']!=8:
        raise RuntimeError('Incomplete short validation; will not start formal training')
    event('short_validated',result=result)
    run('train',formal)
    write('workflow_complete.json',dict(time=time.time(),status='complete'))
except BaseException as e:
    event('stopped',reason=str(e))
    write('workflow_error.json',dict(time=time.time(),error=str(e)))
    raise
