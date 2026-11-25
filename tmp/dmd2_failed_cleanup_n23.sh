set -eu
R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
"$R/env/bin/python" - <<'PY'
from pathlib import Path
import json,shutil
r=Path('/data1/shenglin.zeng/dmd2_official_recipe_20261121').resolve()
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit():continue
 try:args=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
 except (OSError,PermissionError):continue
 assert not any(str(r/'code'/f) in args for f in ['train_matched.py','run_guard.py']),'Own job still active'
targets=[r/'runs'/name/'eval' for name in ['formal_g160','smoke_micro4','smoke_micro2']]
targets.append(r/'tmp')
records=[]
for p in targets:
 if not p.exists():continue
 assert p.resolve().is_relative_to(r) and not p.is_symlink()
 files=[x for x in p.rglob('*') if x.is_file()]
 records.append(dict(path=str(p),files=len(files),bytes=sum(x.stat().st_size for x in files)))
 shutil.rmtree(p)
p=r/'runs/formal_g160/status.json';s=json.loads(p.read_text());s.update(status='failed_oom',failure_time_beijing='2026-11-22 19:38:12',reason='fake backward OOM; own process 46.21GiB plus additional process 1.04GiB; no checkpoint saved')
p.write_text(json.dumps(s,indent=2))
record={'date':'2026-11-23','scope':'Only failed official-recipe group intermediate eval directories and own tmp. Logs/config/code retained. Old successful experiments untouched.',
 'backup':'G0/G104 sixteen sample images plus logs/config copied by rclone and SHA256 verified to local interrupted_g106_20261123 before deletion; no model weights existed.',
 'deleted':records,'total_bytes':sum(x['bytes'] for x in records)}
(r/'失败中间文件清理_20261123.txt').write_text(json.dumps(record,ensure_ascii=False,indent=2))
print(json.dumps(record,ensure_ascii=False,indent=2))
PY
