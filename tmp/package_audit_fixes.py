import ast
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parents[1]
src=root/'sdxl_research'
for p in src.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'), filename=str(p))
files=[p for p in src.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.name!='source_manifest.json']
manifest={p.relative_to(src).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
(src/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
with tarfile.open(root/'tmp/sdxl_audit_fixes_20261022.tar.gz','w:gz') as archive:
    for p in files+[src/'source_manifest.json']:
        archive.add(p,arcname='sdxl_research/'+p.relative_to(src).as_posix())
    archive.add(root/'sdxl_parallel/batching.py',arcname='sdxl_parallel/batching.py')
print('PACKAGED',len(manifest),'source files')
