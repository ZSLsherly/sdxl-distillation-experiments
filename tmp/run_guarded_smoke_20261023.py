"""One bounded probe. Terminate only this subprocess tree on resource collision."""
import json,os,signal,subprocess,sys,time
from pathlib import Path

root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
source=root/'validation/modules_prepared_20261022/sdxl_research'
run='gpu_smoke_20261023/default_b8_g4'
monitor=root/'outputs/gpu_smoke_20261023'
monitor.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(source))
from launch import gpu_check
cards=gpu_check([0,2],36)
uuids={r[1] for r in cards}
config=json.loads((source/'configs/modules/baseline.json').read_text())
config.update(batch=8,micro_batch=4,rollout_batch=4,offload_student_optimizer=True,
              benchmark=True,benchmark_batch=8,benchmark_cycles=4,benchmark_warmup=1)
cfg=monitor/'smoke_config.json';cfg.write_text(json.dumps(config,indent=2))
env=os.environ.copy();env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# The inspected A100 indices are PCI-ordered; launcher rechecks them immediately.
cmd=[sys.executable,str(source/'launch.py'),'--root',str(root),'--gpus','0,2',
     '--config',str(cfg),'--run',run,'--execute']

def descendants(parent):
    children={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                row=(p/'stat').read_text().rsplit(')',1)[1].split()
                children.setdefault(int(row[1]),[]).append(int(p.name))
            except (OSError,ValueError):pass
    found={parent};todo=[parent]
    while todo:
        for p in children.get(todo.pop(),[]):
            if p not in found:found.add(p);todo.append(p)
    return found

started=time.time();reason=None
with (monitor/'launcher.log').open('w') as log, (monitor/'resource_monitor.jsonl').open('w') as metrics:
    proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    (monitor/'launcher.pid').write_text(str(proc.pid))
    while proc.poll() is None:
        try:
            apps=subprocess.check_output(['nvidia-smi','-i','0,2','--query-compute-apps=gpu_uuid,pid,used_memory',
                                          '--format=csv,noheader,nounits'],text=True,timeout=8)
            gpus=subprocess.check_output(['nvidia-smi','-i','0,2','--query-gpu=index,memory.used,memory.free,utilization.gpu',
                                          '--format=csv,noheader,nounits'],text=True,timeout=8)
            owned=descendants(proc.pid)
            foreign=[]
            for line in apps.splitlines():
                row=[x.strip() for x in line.split(',')]
                if len(row)>=2 and row[0] in uuids and int(row[1]) not in owned:
                    foreign.append(int(row[1]))
            metrics.write(json.dumps({'time':time.time(),'gpus':gpus,'apps':apps,'own_pids':sorted(owned),'foreign':foreign})+'\n')
            metrics.flush()
            if foreign:reason='another process entered selected GPUs: '+str(foreign)
        except (subprocess.SubprocessError,OSError,ValueError) as error:
            reason='resource monitoring failed: '+str(error)
        if time.time()-started>1200:reason='20-minute probe timeout'
        if reason:
            # Enumerate only our live descendants; never signal foreign GPU PIDs.
            owned=descendants(proc.pid)
            for pid in sorted(owned-{proc.pid},reverse=True):
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                for pid in descendants(proc.pid):
                    try:os.kill(pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                proc.wait()
            break
        time.sleep(5)
    rc=proc.wait()
result={'returncode':rc,'stop_reason':reason,'wall_seconds':time.time()-started,
        'physical_gpus':[0,2],'initial_gpu_check':cards,'run':run,'config':config}
(monitor/'guard_result.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
sys.exit(rc if rc else (1 if reason else 0))
