set -eu
. /root/sdxl_research/env.sh
python - <<'PY'
from pathlib import Path
import os,re,signal,json
o=Path(os.environ['SDXL_ROOT'])/'outputs/c_mix1_dm234_20261103'
pid=int(re.findall(r'^STARTED (\d+)$',(o/'queue.log').read_text(),re.M)[-1])
p=Path('/proc')/str(pid)
if p.exists():
    argv=p.joinpath('cmdline').read_bytes().split(b'\0')
    assert b'c_mix1_dm234_20261103/model' in argv and b'torch.distributed.run' in argv and os.getpgid(pid)==pid
    os.killpg(pid,signal.SIGTERM)
(o/'SUPERSEDED.txt').write_text('Replaced by fresh c_mix1_dm234_fake3_20261103. Do not resume this two-fake trial.\n')
PY
