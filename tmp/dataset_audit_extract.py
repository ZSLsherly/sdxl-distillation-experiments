from pathlib import Path
from pypdf import PdfReader
import json,re

root=Path(r'D:\study\论文集\推理加速')
out=Path('tmp/dataset_audit_20261021')
out.mkdir(parents=True,exist_ok=True)
manifest=[]
for i,f in enumerate(sorted(root.rglob('*.pdf')),1):
    pages=[p.extract_text(extraction_mode='layout') for p in PdfReader(f).pages]
    key=f'{i:02d}'
    (out/f'{key}.txt').write_text('\n\n'.join(f'=== PDF PAGE {n} ===\n{t}' for n,t in enumerate(pages,1)),encoding='utf-8')
    (out/f'{key}.json').write_text(json.dumps(pages,ensure_ascii=False),encoding='utf-8')
    manifest.append(dict(id=key,name=f.name,path=str(f),pages=len(pages)))
    print(key,len(pages),f.name)
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
