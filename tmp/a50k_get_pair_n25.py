from pathlib import Path
from a50k_stage_assets_n23 import ENV,RCLONE
import subprocess,hashlib
p=Path('output/sdxl-crossdistill/a50k_train60_20261124/G616_holdout.jpg')
src='/data5/shenglin.zeng/a50k_20261124/reports/pair_g616/G616_holdout.jpg'
subprocess.run([RCLONE,'copyto','source:'+src,str(p)],env=ENV,check=True)
h=hashlib.sha256(p.read_bytes()).hexdigest()
assert h=='06856d34a336b0bf64f0e0b00d6eba2d9a8345a7a1bc0ba39dc6131a1f1343fb'
with (p.parent/'训练记录.txt').open('a',encoding='utf-8') as f:
 f.write(f'\n11月25日17:12，北京时间。基线G784/F1568完成，4份归档已验证；消融G616/F1233运行，20k、30k归档通过。G597–616平均 PCM/末步DM/梯度/墙钟：A 0.001104/0.221339/0.186674/62.15秒，无中段 0.000478/0.209575/0.102948/77.10秒。无完整COCO分数。配对G616四条留出同ID、同提示词、seed102。rclone 172.18.33.60:{src} -> Windows:{p.resolve()} SHA256 {h} verified\n')
