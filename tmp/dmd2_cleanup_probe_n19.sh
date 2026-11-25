set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
python3 - <<'PY'
import json,shutil,datetime
from pathlib import Path
r=Path('/data1/shenglin.zeng/dmd2_matched_10k_20261119').resolve()
for proc in Path('/proc').glob('[0-9]*/cmdline'):
 try:args=proc.read_bytes().split(b'\0')
 except (PermissionError,FileNotFoundError,ProcessLookupError):continue
 assert str(r/'code/train_matched.py').encode() not in args, 'Training is active, refuse cleanup'
paths=[p for p in (r/'runs/formal_g160/eval').iterdir() if p.is_dir() and p.name not in {'g000000','g000052'}]
paths += list(r.glob('failed_*/smoke_micro16/eval'))
paths += list((r/'tmp').iterdir())
records=[]
for p in paths:
 assert not p.is_symlink() and p.resolve().is_relative_to(r) and p.resolve()!=r
 files=[q for q in p.rglob('*') if q.is_file()] if p.is_dir() else [p]
 records.append({'path':str(p),'files':len(files),'bytes':sum(q.stat().st_size for q in files)})
 if p.is_dir():shutil.rmtree(p)
 else:p.unlink()
v={'time':datetime.datetime.now().isoformat(),'host':'172.18.132.41','preserved':'configs, logs, metrics, G0 and latest G52 preview; no saved model checkpoint existed','removed':records,'bytes':sum(x['bytes'] for x in records)}
(r/'清理记录_20261119.txt').write_text(json.dumps(v,ensure_ascii=False,indent=2))
print(json.dumps({'removed_MiB':v['bytes']/2**20,'paths':len(paths),'files':sum(x['files'] for x in records)}))
PY
export CUDA_VISIBLE_DEVICES=GPU-3a1e6a0e-9fab-18ef-c543-30066e01e383,GPU-dee80cca-d29a-9086-d006-b76ba6ee093e,GPU-4fba4d7f-66e5-aa77-7210-3803fe7190df,GPU-baca962f-0eff-b198-1677-8c28d525ecc1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 OMP_NUM_THREADS=2
nvidia-smi --id="$CUDA_VISIBLE_DEVICES" --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits
"$R/env/bin/python" -m torch.distributed.run --standalone --nproc_per_node=4 "$R/code/gpu_probe.py"
