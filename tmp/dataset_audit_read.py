from pathlib import Path
import json,re,sys
base=Path('tmp/dataset_audit_20261021')
manifest=json.loads((base/'manifest.json').read_text(encoding='utf-8'))
ids=sys.argv[1].split(',')
pattern=sys.argv[2] if len(sys.argv)>2 else r'dataset|training data|LAION|Journey|OpenVid|Vript|WebVid|VidProM|synthetic|COYO|DataComp'
for item in manifest:
    if item['id'] not in ids: continue
    print('\n###',item['id'],item['name'])
    pages=json.loads((base/(item['id']+'.json')).read_text(encoding='utf-8'))
    if pattern.startswith('pages='):
        for p in map(int,pattern[6:].split(',')): print(f'--- PDF PAGE {p} ---\n'+re.sub(r'[ \t]{2,}',' ',pages[p-1]))
        continue
    for p,t in enumerate(pages,1):
        lines=t.splitlines(); hits=set()
        for i,l in enumerate(lines):
            if re.search(pattern,l,re.I): hits.update(range(max(0,i-2),min(len(lines),i+4)))
        if hits:
            print(f'--- PDF PAGE {p} ---')
            prev=-2
            for i in sorted(hits):
                if i!=prev+1: print('...')
                print(lines[i]);prev=i
