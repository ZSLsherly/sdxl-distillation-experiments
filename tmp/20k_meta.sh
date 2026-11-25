set -eu
. /root/sdxl_research/env.sh
for f in /sys/fs/cgroup/memory/memory.limit_in_bytes /sys/fs/cgroup/memory/memory.usage_in_bytes /sys/fs/cgroup/memory/memory.stat /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us ; do test ! -e "$f" || { echo "$f"; head -4 "$f"; }; done
grep -m1 -E '^model name' /proc/cpuinfo
ls "$SDXL_WORK/bc_texture_inputs/models/sdxl-base-1.0"
df -B1 --output=avail,target "$SDXL_WORK"
python - <<'PY'
import requests
rev='be22767697a1f3ca656b73c776e15fa335c86c6c'; folder='data/laion_vae_latents/sdxl_vae_latents_laion_500k'
for uri in [f'https://hf-mirror.com/api/models/tianweiy/DMD2/tree/{rev}/{folder}']:
 r=requests.get(uri,timeout=30);r.raise_for_status()
 for v in r.json():
  if v['path'].endswith(('002.pt','003.pt')): print(v)
PY
