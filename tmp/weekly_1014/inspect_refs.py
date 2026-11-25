import zipfile,xml.etree.ElementTree as E,json,collections,os
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
for f in ['D:/study/论文集/推理加速/周报/周报0916.pptx','D:/study/论文集/推理加速/周报/周报.pptx']:
 z=zipfile.ZipFile(f); p=E.fromstring(z.read('ppt/presentation.xml'));print('\nFILE',f,'SIZE',p.find('p:sldSz',ns).attrib)
 fonts=collections.Counter();sizes=collections.Counter();colors=collections.Counter()
 for name in sorted([n for n in z.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')],key=lambda n:int(n.rsplit('slide',1)[1].split('.')[0])):
  root=E.fromstring(z.read(name));txt=[x.text for x in root.findall('.//a:t',ns)];print(name,' / '.join(txt)[:2200])
  for a in root.findall('.//a:rPr',ns):
   if 'sz' in a.attrib:sizes[a.attrib['sz']]+=1
  for a in root.findall('.//a:latin',ns)+root.findall('.//a:ea',ns):fonts[a.attrib.get('typeface')]+=1
  for a in root.findall('.//a:srgbClr',ns):colors[a.attrib.get('val')]+=1
 print('FONTS',fonts,'SIZES',sizes,'COLORS',colors)
