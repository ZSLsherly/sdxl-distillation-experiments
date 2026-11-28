import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from launch import command, gpu_check

code = Path(__file__).resolve().parent
root = code.parent
lock = (root / 'run.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
dataset = 'parallel_50k_seed20261018'
if not (root/'datasets/已核验传输.txt').is_file():
    raise RuntimeError('Data transfer verification is still pending')
base = json.loads((code/'base.json').read_text())
base.update(data_dir='datasets/'+dataset, checkpoint_mode='full', save_every_updates=0,
            snapshot_seconds=10**8, hours=168)
gpus = [0,1,2,3]
logs = root/'logs'
logs.mkdir(exist_ok=True)
env = dict(os.environ, CUDA_VISIBLE_DEVICES='0,1,2,3', NCCL_IB_DISABLE='1', NCCL_P2P_DISABLE='1',
           OMP_NUM_THREADS='4', HF_HUB_OFFLINE='1', TORCHDYNAMO_DISABLE='1', PYTHONUNBUFFERED='1')
for key in ['TMPDIR','TORCH_HOME','HF_HOME','TRITON_CACHE_DIR','CUDA_CACHE_PATH','XDG_CACHE_HOME']:
    path=root/'cache'/key
    path.mkdir(parents=True,exist_ok=True)
    env[key]=str(path)
def run(name,config):
    if (root/'outputs'/name).exists():raise FileExistsError(name)
    while True:
        try:
            gpu_check(gpus,36)
            break
        except RuntimeError as exc:
            if 'another compute task' not in str(exc) and 'insufficient headroom' not in str(exc):raise
            print('WAIT_GPU',name,str(exc),flush=True)
            time.sleep(60)
    with (logs/(name+'.log')).open('x') as log:
        return subprocess.call(command(config,root,name,4),env=env,stdout=log,stderr=subprocess.STDOUT)

base.update(micro_batch=16,max_student_updates=784)
if shutil.disk_usage(root).free < 255*1024**3:raise RuntimeError("Insufficient space for four resumable states")
raise SystemExit(run("A50K_clean_x0",base))
