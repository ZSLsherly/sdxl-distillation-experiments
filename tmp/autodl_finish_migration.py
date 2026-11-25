import json,os,shutil
from pathlib import Path
src=Path('/root/autodl-tmp/sdxl_research').resolve();dst=Path('/root/sdxl_research').resolve()
assert src==Path('/root/autodl-tmp/sdxl_research') and dst==Path('/root/sdxl_research')
state=json.loads((dst/'storage_layout.json').read_text())
assert state['status']=='system_migration_cpu_verified' and state['copy_checksum_verified']
# Scripts installed before the move may retain an original interpreter shebang.
fixed=[]
for p in (dst/'env/bin').iterdir():
 if not p.is_file() or p.is_symlink():continue
 with p.open('rb') as f:first=f.read(256)
 if first.startswith(b'#!') and b'python' in first.split(b'\n')[0]:
  content=p.read_bytes();line,sep,rest=content.partition(b'\n')
  desired=b'#!/root/sdxl_research/env/bin/python'
  if line!=desired:p.write_bytes(desired+sep+rest);fixed.append(p.name)
# Complete the verified move; retain old paths as compatibility symlinks.
for name in ['env','runtime','models','datasets','code','conditions.pt','prompts.json']:
 old=src/name;new=dst/name
 assert new.exists() and new.resolve().is_relative_to(dst)
 if old.is_symlink():
  assert old.resolve()==new.resolve();continue
 assert old.resolve().is_relative_to(src)
 if old.is_dir():shutil.rmtree(old)
 else:old.unlink()
 old.symlink_to(new,target_is_directory=new.is_dir())
state.update(legacy_paths='compatibility symlinks to system disk',fixed_entrypoints=fixed,system_free_GiB=shutil.disk_usage(dst).free/1024**3,data_free_GiB=shutil.disk_usage(src).free/1024**3)
(dst/'storage_layout.json').write_text(json.dumps(state,indent=2))
print(json.dumps(state),flush=True)
