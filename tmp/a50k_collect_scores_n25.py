"""Collect the finished small result artifacts via rclone and verify both endpoints."""
from pathlib import Path
import time,json,hashlib,subprocess
from a50k_stage_assets_n23 import RCLONE,ENV,remote
R='/data/shenglin.zeng/a50k_20261123/eval_A_milestones_20261125'
local=Path('output/sdxl-crossdistill/a50k_train60_20261124/coco_milestones')
def get(rel):
 dest=local/rel
 dest.parent.mkdir(parents=True,exist_ok=True)
 subprocess.run([RCLONE,'copyto','dest:'+R+'/'+rel,str(dest)],env=ENV,check=True)
 h=hashlib.sha256(dest.read_bytes()).hexdigest()
 assert h==remote('sha256sum '+R+'/'+rel).split()[0]
 with (local/'取回校验.txt').open('a',encoding='utf8') as f:f.write(f'172.18.32.81:{R}/{rel} -> {dest.resolve()} SHA256 {h}\n')
 return json.loads(dest.read_text(encoding='utf8'))
while True:
 try:
  if remote('test -f '+R+'/summary.json && echo ready || true')=='ready':break
 except (OSError,RuntimeError):pass
 time.sleep(180)
scores=get('summary.json')
rows=['A基线单遍50K：各学生检查点，同协议COCO10K/CLIP1024实测','样本数\tG\tFID↓\tPatch FID↓\tCLIP×100↑']
for n,g in [(20000,313),(30000,470),(40000,627),(50000,784)]:
 arm=f'A{n}';v=get(f'results/{arm}/all_complete.json');assert v==scores[arm]
 rows.append(f"{n}\t{g}\t{v['FID']:.4f}\t{v['Patch_FID']:.4f}\t{v['CLIP']:.4f}")
(local/'实测结果.txt').write_text('\n'.join(rows)+'\n',encoding='utf8')
print('\n'.join(rows))
