"""CPU-only split/prompt-overlap audit. Reports evidence; does not silently change data."""
import argparse,csv,hashlib,io,json,zipfile
from pathlib import Path
import torch

def canonical(text):return ' '.join(text.casefold().split())
def audit(prepared,coco_zip):
    data=torch.load(prepared,weights_only=True,mmap=True,map_location='cpu')
    n=data['train_count'];prompts=list(map(canonical,data['prompts']))
    train=set(prompts[:n]);heldout=set(prompts[n:])
    with zipfile.ZipFile(coco_zip) as z:
        raw=z.read('coco10k/subset.csv')
        coco=[canonical(r['caption']) for r in csv.DictReader(io.StringIO(raw.decode()))]
    return {'train_count':n,'holdout_count':len(prompts)-n,
        'unique_train_prompts':len(train),'train_holdout_exact_prompt_overlap':len(train&heldout),
        'COCO10K_overlapping_rows':sum(x in train for x in coco),
        'COCO10K_overlapping_unique_prompts':len(train&set(coco)),
        'train_prompt_order_sha256':hashlib.sha256(json.dumps(prompts[:n],ensure_ascii=False).encode()).hexdigest(),
        'coco_csv_sha256':hashlib.sha256(raw).hexdigest(),
        'limitation':'Exact normalized text only; does not exclude semantic/image near duplicates or teacher pretraining overlap'}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepared',type=Path,required=True)
    ap.add_argument('--coco-zip',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();r=audit(a.prepared,a.coco_zip);a.out.write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
