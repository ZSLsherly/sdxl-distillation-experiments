set -eu
work=/tmp/laion-discovery-904.b9IQDB
repo=Ashly/LAION-10K
ctl="$HOME/.discovery/bin/discovery-ctl"
"$ctl" --config-dir "$work/config" dataset list --dataset "$repo" --version main --format json > "$work/main.json"
"$ctl" --config-dir "$work/config" dataset download-file --dataset "$repo" --remote-path /manifest.json --output "$work/check" --format json > "$work/check.json"
python3 - <<'PY'
from pathlib import Path
import json
w=Path('/tmp/laion-discovery-904.b9IQDB')
s=Path('/data5/shenglin.zeng/sdxl_crossdistill/datasets/parallel_10k_seed20261018')
m=list((w/'check').rglob('manifest.json'))
assert len(m)==1 and m[0].read_bytes()==(s/'manifest.json').read_bytes()
print('MANIFEST_BYTE_MATCH',m[0].stat().st_size)
PY
