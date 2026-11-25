python3 - <<'PY'
from pathlib import Path
import os,signal,time
name='A_clean_global_high30_g160_20261114_v2';pid=931847;child=931936
cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes()
assert ("GROUP='"+name+"'").encode() in cmd and b'fzrange16_' in cmd
cc=(Path('/proc')/str(child)/'cmdline').read_bytes().split(b'\0')[0]
assert cc.endswith(b'/fzsftp')
st=(Path('/proc')/str(child)/'status').read_text()
assert int(next(x.split()[1] for x in st.splitlines() if x.startswith('PPid:')))==pid
root=Path('/data5/shenglin.zeng/checkpoints/a800_archive_20261114/outputs')/name/'model/weights/20261114T172912+0800_g000160_end'
size=5135506134
missing=[i for i in range(16) if (root/f'student_rank0.pt.fzrange16_{i}').stat().st_size<size*(i+1)//16]
assert missing==[14],missing
assert (root/'student_rank0.pt.fzrange16_14').stat().st_size==size*14//16
os.kill(pid,signal.SIGTERM);os.kill(child,signal.SIGTERM)
time.sleep(2)
print('STOPPED_ONLY_OWN_STALLED_TRANSFER',pid,child,'PRESERVED_15_COMPLETE_RANGES')
PY
