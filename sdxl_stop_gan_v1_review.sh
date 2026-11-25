source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import os,signal,time,json,hashlib
from pathlib import Path
root=Path.cwd();selected={};roots=[]
def info(pid):
    p=Path('/proc')/str(pid)
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode().strip()
        status=(p/'status').read_text();ppid=int(next(x.split()[1] for x in status.splitlines() if x.startswith('PPid:')))
        return {'pid':pid,'ppid':ppid,'cmd':cmd,'start':(p/'stat').read_text().split()[21],'uid':p.stat().st_uid,'cwd':os.readlink(p/'cwd')}
    except (FileNotFoundError,ProcessLookupError,PermissionError):return None
for marker,script in [('gan_stage_pilot','sdxl_gan_runner.sh'),('gan_assessment','sdxl_gan_finish.sh')]:
    path=root/'logs'/f'{marker}.pid'
    if path.exists():
        pid=int(path.read_text());i=info(pid)
        if i:
            assert i['uid']==os.getuid() and i['cwd']==str(root) and f'bash src/{script}' in i['cmd'],i
            selected[pid]=i;roots.append(pid)
all_processes={}
for p in Path('/proc').iterdir():
    if p.name.isdigit():
        i=info(int(p.name))
        if i:all_processes[i['pid']]=i
for _ in range(20):
    added={p:i for p,i in all_processes.items() if i['ppid'] in selected and p not in selected}
    if not added:break
    assert all(i['uid']==os.getuid() for i in added.values())
    selected.update(added)
out=root/'outputs/gan_stage_pilot';out.mkdir(exist_ok=True)
record={'status':'stopped_for_protocol_revision','reason':'Verified review R1/R2/R3; stage-matched adversary and clean efficiency control required. No v1 quality ranking.','processes':list(selected.values()),'sources':{p:hashlib.sha256((root/'src'/p).read_bytes()).hexdigest() for p in ['sdxl_gan_train.py','sdxl_pixel_gan.py','sdxl_objectives.py','sdxl_gan_runner.sh']}}
(out/'review_stop.json').write_text(json.dumps(record,indent=2))
def send(pid,sig):
    current=info(pid)
    if current and current['start']==selected[pid]['start']:
        try:os.kill(pid,sig)
        except ProcessLookupError:pass
for pid in roots:send(pid,signal.SIGSTOP)
for pid in sorted(selected,reverse=True):
    if pid not in roots:send(pid,signal.SIGTERM)
for pid in roots:send(pid,signal.SIGTERM);send(pid,signal.SIGCONT)
time.sleep(4)
for pid in selected:send(pid,signal.SIGKILL)
remaining=[p for p in selected if (i:=info(p)) and i['start']==selected[p]['start']]
record['remaining_verified_processes']=remaining
(out/'review_stop.json').write_text(json.dumps(record,indent=2))
print(json.dumps({'stopped_pids':list(selected),'remaining':remaining,'source_hashes':record['sources']},indent=2))
PY
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
