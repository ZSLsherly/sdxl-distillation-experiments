set -e
source /data5/shenglin.zeng/env.sh
python - <<'PY'
from pathlib import Path
r=Path('/data5/shenglin.zeng')
print((r/'outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/config.yaml').read_text())
print('NEGATIVE_EMBED_BYTES',(r/'models/negative_embed.pt').stat().st_size)
for p in (r/'cache/huggingface').glob('*/*'):print('HF_CACHE',str(p),str(p.resolve()))
print('VIDEO_PROCESSES')
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  args=(p/'cmdline').read_bytes().decode(errors='replace').split('\0')
  if args and any(x in args[0] for x in ['python','torchrun']):
   cmd=' '.join(args)
   if any(x in cmd for x in ['Wan2.1','CrossDistill_Smoke','SparkDiffusion']):print(p.name,cmd)
 except (FileNotFoundError,PermissionError):pass
PY
