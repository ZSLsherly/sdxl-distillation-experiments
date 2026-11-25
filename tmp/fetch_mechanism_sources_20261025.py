from pathlib import Path
from html.parser import HTMLParser
import urllib.request,concurrent.futures,json,re,hashlib,sys,subprocess
root=Path(__file__).resolve().parents[1]
out=root/'output/sdxl-crossdistill/mechanism_research_20261025/sources';out.mkdir(parents=True,exist_ok=True)
sources={
 'dymd':'https://arxiv.org/html/2609.31349v1',
 'crossdistill':'https://arxiv.org/html/2609.14725v1',
 'dmd2':'https://arxiv.org/html/2405.14867v1',
 'flash_dmd_arxiv':'https://arxiv.org/html/2511.20549v1',
 'phased_dmd':'https://arxiv.org/html/2510.27684v1',
 'tdm':'https://arxiv.org/html/2503.06674v1',
 'rtdmd':'https://arxiv.org/html/2605.26108v1',
 'rcm':'https://arxiv.org/html/2510.08431v1',
 'cdm':'https://arxiv.org/html/2605.06376v1',
 'flash_dmd_cvpr':'https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Flash-DMD_Towards_High-Fidelity_Few-Step_Image_Generation_with_Efficient_Distillation_and_CVPR_2026_paper.pdf',
 'flash_dmd_cvpr_supp':'https://openaccess.thecvf.com/content/CVPR2026/supplemental/Chen_Flash-DMD_Towards_High-Fidelity_CVPR_2026_supplemental.pdf'}
class Text(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.math=0;self.skip=0
 def handle_starttag(self,tag,attrs):
  a=dict(attrs)
  if tag in ('script','style'):self.skip+=1
  if tag=='math':
   if not self.math:self.parts.append(' $'+a.get('alttext','')+'$ ')
   self.math+=1
  elif not self.math and not self.skip:
   if tag in ('p','tr','h1','h2','h3','h4','h5','li','section','div'):self.parts.append('\n')
   if tag in ('td','th'):self.parts.append(' | ')
   if tag=='a' and a.get('href','').startswith('https://github.com/'):self.parts.append(' ['+a['href']+'] ')
 def handle_endtag(self,tag):
  if tag=='math':self.math=max(0,self.math-1)
  if tag in ('script','style'):self.skip=max(0,self.skip-1)
  if not self.math and tag in ('p','tr','h1','h2','h3','h4','h5','li'):self.parts.append('\n')
 def handle_data(self,s):
  if not self.math and not self.skip:self.parts.append(s)
def one(item):
 name,url=item
 try:
  data=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0'}),timeout=60).read()
  ext='.pdf' if url.endswith('.pdf') else '.html';p=out/(name+ext);p.write_bytes(data)
  if ext=='.pdf':
   target=out/(name+'.txt')
   from pypdf import PdfReader
   text='\n'.join(f'\n=== PAGE {i+1} ===\n'+page.extract_text() for i,page in enumerate(PdfReader(p).pages)).replace('\x00','')
  else:
   parser=Text();parser.feed(data.decode('utf-8'));text='\n'.join(re.sub(r'\s+',' ',s).strip() for s in ''.join(parser.parts).splitlines() if s.strip())
  (out/(name+'.txt')).write_text(text,encoding='utf-8')
  return {'name':name,'url':url,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'text_chars':len(text)}
 except Exception as e:return {'name':name,'url':url,'error':str(e)}
if __name__=='__main__':
 with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
  results=list(pool.map(one,sources.items()))
 (out/'manifest.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
 for r in results:print(json.dumps(r),flush=True)
