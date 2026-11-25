"""Prepare and queue the paired 50K ablation without replacing the active trainer."""
import hashlib
import json
import shlex
import subprocess
import sys
from pathlib import Path
from a50k_stage_assets_n23 import RCLONE, ENV, remote

host='172.18.33.60'
root='/data5/shenglin.zeng/a50k_20261124'
local=Path('output/sdxl-crossdistill/a50k_train60_20261124')
dest=local/'no_middle'
dest.mkdir(exist_ok=True)
name='A50K_no_middle_singlepass'
base=(local/'deploy/queue50.py').read_text(encoding='utf-8')
wait="""follow_lock=(root/'no_middle_queue.lock').open('a')
fcntl.flock(follow_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
print('WAIT_FOR_BASELINE G784 F1568 samples=50000',flush=True)
while True:
    status_path=root/'outputs/A50K_singlepass/status.json'
    if status_path.is_file():
        status=json.loads(status_path.read_text())
        if status.get('status')=='complete':
            assert (status['student_updates'],status['fake_updates'],status['student_samples'])==(784,1568,50000)
            marker=Path(status['training_state'])/'complete.json'
            assert json.loads(marker.read_text())['progress']['student_samples']==50000
            break
    time.sleep(60)
lock=(root/'run.lock').open('a')
fcntl.flock(lock,fcntl.LOCK_EX)
"""
a=base.index('lock = ');b=base.index("dataset = ",a)
script=base[:a]+wait+base[b:]
storage='/data6/shenglin.zeng/a50k_20261124/no_middle/training_state'
script=script.replace("    with (logs/(name+'.log')).open('x') as log:",f"""    out=root/'outputs'/name
    out.mkdir()
    (out/'training_state').symlink_to({storage!r},target_is_directory=True)
    print('START',name,'terminal_only_dm=True',flush=True)
    with (logs/(name+'.log')).open('x') as log:""")
script=script.replace('base.update(micro_batch=16,max_student_updates=784)','base.update(micro_batch=16,max_student_updates=784,terminal_only_dm=True)').replace('shutil.disk_usage(root)','shutil.disk_usage('+repr(storage)+')').replace('run("A50K_singlepass",base)',f'run("{name}",base)')
compile(script,'after_singlepass_no_middle.py','exec')
p=dest/'after_singlepass_no_middle.py'
p.write_bytes(script.encode())
source=local/'deploy'
remote_checks=remote('cd '+root+'/code && sha256sum queue50.py train.py boundary.py base.json launch.py',host)
for line in remote_checks.splitlines():
    checksum,file=line.split()
    assert hashlib.sha256((source/file).read_bytes()).hexdigest()==checksum,file
target=root+'/code/'+p.name
subprocess.run([RCLONE,'copyto',str(p),'source:'+target],env=ENV,check=True)
checksum=hashlib.sha256(p.read_bytes()).hexdigest()
assert remote('sha256sum '+target,host).split()[0]==checksum
print(remote('set -eu\nmkdir -p '+storage+'\ntest -w '+storage+'\ncd '+root+
    '\nexport TMPDIR="$PWD/cache/TMPDIR" OMP_NUM_THREADS=4\npy=/data5/shenglin.zeng/envs/crossdistill/bin/python\n"$py" -m py_compile code/'+p.name+
    '\nnohup "$py" -u code/'+p.name+' >>logs/no_middle_queue.log 2>&1 </dev/null &\nworker=$!\nsleep 2\nkill -0 "$worker"\necho "PID=$worker"\ntail -3 logs/no_middle_queue.log',host))
(dest/'排队与校验.txt').write_text(f'源 {p.resolve()}\n目标 {host}:{target}\nrclone SHA256 {checksum} 两端一致。\n'
    f'前序 A50K_singlepass G784/F1568、50,000样本及最终可续训存档完成后启动 {name}。全新SDXL初始化。'
    '训练参数唯一变化为 terminal_only_dm=True。学生PCM+末步DM，fake维持配对中段及末步；其余沿用单遍基线，包括噪声种子、评估和存档。'
    f'\n本机存档 {host}:{storage}。源代码校验如下：\n'+remote_checks,encoding='utf-8')
