"""Read saved evaluation PNGs only; create labeled comparison figures locally."""
import io
import json
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / '.image_runtime_pkgs'))
import paramiko
from PIL import Image, ImageDraw, ImageFont

raw = Path(r'C:\Users\sherl\Desktop\login.txt').read_text(encoding='utf-8-sig')
user, password = next((u, p.strip()) for ip, u, p in re.findall(
    r'IP:\s*(\d+\.\d+\.\d+\.\d+)\s+User:\s*(\S+)\s+Password:\s*([^\r\n]+)', raw)
    if ip == '172.18.33.60')
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('172.18.33.60', username=user, password=password, timeout=15,
               look_for_keys=False, allow_agent=False)
remote = r'''
import io,json,re,sys,zipfile
from datetime import datetime
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs')
runs=[('dmd_only',root/'dmd_control/dmd_only_20261018T141942_4gpu_mb16'),('joint4',root/'gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16'),
      ('gan4',root/'gan_stage_parallel/gan4_20261017T232317_mb16'),
      ('joint34',root/'gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16')]
evaluations={}
for key,p in runs:
    config=json.loads((p/'config.json').read_text())
    world=config['world_size']
    evaluations[key]={}
    for ev in sorted((p/'eval').iterdir()):
        if not all((ev/f'rank{rank}.json').exists() for rank in range(world)): continue
        samples=[]
        for rank in range(world): samples+=json.loads((ev/f'rank{rank}.json').read_text())['samples']
        if not all((ev/s['image']).exists() for s in samples): continue
        step=int(re.search(r'_g(\d+)_',ev.name).group(1))
        evaluations[key][step]=(ev,samples)
common=160
assert all(common in v for v in evaluations.values())
identity=None
for key in evaluations:
    for step in [common]:
        _,samples=evaluations[key][step]
        ids=sorted((s['split'],s['original_index'],s['prompt'],s['seed']) for s in samples)
        if identity is None:identity=ids
        assert ids==identity,(key,step,'Evaluation prompt/seed mismatch')
manifest={'retrieved_at':datetime.now().astimezone().isoformat(),'common_step':common,'groups':{}}
buf=io.BytesIO()
with zipfile.ZipFile(buf,'w',compression=zipfile.ZIP_STORED) as z:
    for group in ['matched']:
        manifest['groups'][group]={}
        for key,p in runs:
            step=max(evaluations[key]) if group=='latest' else common
            ev,samples=evaluations[key][step]
            manifest['groups'][group][key]={'step':step,'eval_dir':str(ev),'samples':samples}
            for s in samples:
                z.write(ev/s['image'],f'{group}/{key}/{s["image"]}')
    z.writestr('manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
sys.stdout.buffer.write(buf.getvalue())
'''
stdin, stdout, stderr = client.exec_command('source /data5/shenglin.zeng/env.sh; python -', timeout=180)
stdin.write(remote)
stdin.flush()
stdin.channel.shutdown_write()
archive = stdout.read()
error = stderr.read().decode()
code = stdout.channel.recv_exit_status()
client.close()
if code:
    raise RuntimeError(error)
bundle = zipfile.ZipFile(io.BytesIO(archive))
manifest = json.loads(bundle.read('manifest.json'))
stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
out = ROOT / 'output/sdxl-crossdistill/benchmark_final_20261020'
out.mkdir(parents=True,exist_ok=True)
(out / 'image_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

font_path = 'C:/Windows/Fonts/msyh.ttc'
font = lambda size: ImageFont.truetype(font_path, size)
labels = [('dmd_only', '纯 DMD（保留首步TM）'),('joint4', '第4步 GAN+DMD'), ('gan4', '第4步纯 GAN'), ('joint34', '第3、4步 GAN+DMD')]
names = {7084:'科幻场景',8109:'培根面饼',5122:'汽车',592:'展览馆',986:'靠垫与餐桌',14585:'婚礼',10099:'飞船',335:'现代住宅'}
colors = ['#8A45B5','#3464a8','#a56627','#31816b']

def load_image(group, key, sample):
    return Image.open(io.BytesIO(bundle.read(f'{group}/{key}/{sample["image"]}'))).convert('RGB')

def draw(group, split, filename, crops=False):
    chosen = [('holdout',986),('holdout',14585),('holdout',10099),('holdout',335)] if crops else [
        (split,idx) for idx in ([7084,8109,5122,592] if split=='train' else [986,14585,10099,335])]
    cell = 384 if crops else 420
    margin, gap, header, row_h = 28, 16, 192, cell+58
    width = margin*2 + cell*4 + gap*3
    canvas = Image.new('RGB',(width,header+len(chosen)*row_h+64),'#f4f6f8')
    d = ImageDraw.Draw(canvas)
    title = '同训练步数｜原生像素纹理对照' if crops else ('最新结果｜留出样本' if split=='holdout' else '最新结果｜训练样本')
    if group=='matched' and not crops: title='同训练步数｜留出样本对照'
    d.text((margin,20),title,font=font(32),fill='#17212e')
    note = f'全部为 G{manifest["common_step"]}；相同提示词、种子102。' if group=='matched' else '各方案最新完整评估；更新次数不同，请勿直接据此判断优劣。'
    d.text((margin,70),note,font=font(19),fill='#516170')
    for col,(key,label) in enumerate(labels):
        x=margin+col*(cell+gap)
        meta=manifest['groups'][group][key]
        d.rounded_rectangle((x,110,x+cell,174),radius=9,fill=colors[col])
        d.text((x+14,116),label,font=font(23),fill='white')
        d.text((x+14,148),f'G{meta["step"]} · 4步推理',font=font(17),fill='white')
        lookup={(s['split'],s['original_index']):s for s in meta['samples']}
        for row,(part,idx) in enumerate(chosen):
            y=header+row*row_h
            s=lookup[(part,idx)]
            im=load_image(group,key,s)
            if crops:
                x0,y0={14585:(320,320),8109:(256,448),986:(256,320),10099:(320,320),335:(320,320)}[idx]
                im=im.crop((x0,y0,x0+384,y0+384))
            else:
                im=im.resize((cell,cell),Image.Resampling.LANCZOS)
            canvas.paste(im,(x,y))
            d.text((x+4,y+cell+6),f'{names[idx]} · {"留出" if part=="holdout" else "训练"} #{idx}',font=font(18),fill='#243444')
    footer='局部为原图384×384裁剪：未锐化、未调色、未放大。' if crops else '整图仅等比例缩小展示；未锐化、未调色。G表示学生优化器更新次数。'
    d.text((margin,canvas.height-47),footer,font=font(18),fill='#516170')
    canvas.save(out/filename)
    return filename

files=[draw('matched','holdout','matched_holdout.png'),draw('matched','holdout','matched_texture.png',True)]
print(json.dumps({'output':str(out),'matched_step':common if 'common' in globals() else manifest['common_step'],'files':files},ensure_ascii=False))
