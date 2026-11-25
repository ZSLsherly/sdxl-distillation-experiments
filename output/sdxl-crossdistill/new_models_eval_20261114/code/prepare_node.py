import json,sys
from pathlib import Path
ROOT=Path(sys.argv[1]);node=json.loads((ROOT/'node.json').read_text())
for local,source in node['links'].items():
 dest=ROOT/local;src=Path(source);assert src.exists(),source
 dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists():dest.symlink_to(src,target_is_directory=src.is_dir())
for arm in node['arms']:
 dest=ROOT/'results'/arm/'features/real';dest.parent.mkdir(parents=True,exist_ok=True)
 if not dest.exists() and not dest.is_symlink():dest.symlink_to(ROOT/'reference/features/real',target_is_directory=True)
print('LOCAL_ASSETS_LINKED',node['host'])
