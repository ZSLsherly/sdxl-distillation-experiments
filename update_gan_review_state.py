from pathlib import Path
import re
p=Path('output/sdxl-crossdistill/gan_stage_protocol.html')
s=p.read_text(encoding='utf-8')
note='<div class="note"><b>v1已停止并被v2取代：</b>独立代码审查确认阶段3判别器分布、效率对照和预检状态问题；旧A组停在6次学生更新。<a href="reviews/gan_review_resolution_20261017.html">查看逐项复核、修订设计与新版预检证据</a>。下文保留原方案作历史记录，当前执行以v2修订说明为准。</div>'
s,n=re.subn(r'<div class="note">.*?</div>',note,s,count=1,flags=re.S);assert n==1
p.write_text(s,encoding='utf-8')
