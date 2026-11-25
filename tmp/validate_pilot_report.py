from pathlib import Path
from html.parser import HTMLParser
root=Path('output/sdxl-crossdistill/hparam_pilot')
class Links(HTMLParser):
 def __init__(self):super().__init__();self.urls=[]
 def handle_starttag(self,tag,attrs):
  self.urls += [v for k,v in attrs if k in ['src','href'] and v and not v.startswith(('http:','https:','#'))]
p=Links();p.feed((root/'report.html').read_text(encoding='utf-8'))
missing=[u for u in p.urls if not (root/u).is_file()]
assert not missing,missing
print('REPORT_LOCAL_LINKS_PASS',len(p.urls))
