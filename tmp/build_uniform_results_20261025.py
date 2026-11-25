from pathlib import Path
import sys,json,tarfile,textwrap,statistics,html
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'.plot_runtime_pkgs'))
from PIL import Image,ImageDraw,ImageFont
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

out=root/'output/sdxl-crossdistill/uniform10k_results_20261025'
with tarfile.open(out/'results.tar.gz') as tar:tar.extractall(out,filter='data')
r=json.loads((out/'results.json').read_text())
font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19)
small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
steps=[0,40,80,120,160]
cases=sorted(r['evals']['0']['samples'],key=lambda x:(x['split'],x['original_index']))
for split in ('holdout','train'):
 selected=[x for x in cases if x['split']==split]
 width=320;row_h=410;top=42
 grid=Image.new('RGB',(width*5,top+row_h*len(selected)),(245,247,250));d=ImageDraw.Draw(grid)
 crops=Image.new('RGB',(256*5,top+300*len(selected)),(245,247,250));dc=ImageDraw.Draw(crops)
 for j,g in enumerate(steps):
  d.text((j*width+8,10),f'G{g}'+(' / original 4-step' if g==0 else ''),font=font,fill=(20,40,60))
  dc.text((j*256+8,10),f'G{g} / center 256px',font=small,fill=(20,40,60))
 for i,case in enumerate(selected):
  prompt=case['prompt'];label=f"{split} #{case['original_index']}: {prompt}"
  d.multiline_text((8,top+i*row_h),'\n'.join(textwrap.wrap(label,140))[:400],font=small,fill=(20,40,60),spacing=3)
  dc.text((8,top+i*300),f"{split} #{case['original_index']}",font=small,fill=(20,40,60))
  for j,g in enumerate(steps):
   p=out/f'images/g{g:03d}'/case['image'];im=Image.open(p).convert('RGB')
   grid.paste(im.resize((width,width),Image.Resampling.LANCZOS),(j*width,top+i*row_h+75))
   w,h=im.size;crops.paste(im.crop((w//2-128,h//2-128,w//2+128,h//2+128)),(j*256,top+i*300+35))
 grid.save(out/f'{split}_progress.jpg',quality=95)
 crops.save(out/f'{split}_center_crops.png')

selected=[x for x in cases if x['split']=='holdout'][:2]
quick=Image.new('RGB',(1024,1152),(245,247,250));d=ImageDraw.Draw(quick)
for j,g in enumerate((0,160)):
 d.text((j*512+12,8),'Original SDXL / same 4-step sampler' if g==0 else 'Trained G160 / GAN+DMD at step4',font=font,fill=(20,40,60))
for i,case in enumerate(selected):
 d.multiline_text((10,40+i*556),'\n'.join(textwrap.wrap(case['prompt'],100)),font=small,fill=(20,40,60))
 for j,g in enumerate((0,160)):
  im=Image.open(out/f'images/g{g:03d}'/case['image']).convert('RGB')
  quick.paste(im.resize((512,512),Image.Resampling.LANCZOS),(j*512,80+i*556))
quick.save(out/'initial_vs_final.jpg',quality=95)

gs=[x for x in r['rows']['0'] if x['phase']=='student_done']
fs=[x for x in r['rows']['0'] if x['phase']=='fake_D']
def smooth(xs,n=7):return [statistics.mean(xs[max(0,i-n+1):i+1]) for i in range(len(xs))]
plt.rcParams.update({'figure.dpi':150,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(14,8))
for stage in (2,3,4):
 a=[x for x in gs if x['stage']==stage]
 axes[0,0].plot([x['g'] for x in a],smooth([x['dmd_loss'] for x in a]),label=f'Stage {stage}')
axes[0,0].set_title('DMD surrogate (7 same-stage updates mean)');axes[0,0].legend()
axes[0,1].plot([x['g'] for x in gs],smooth([x['pcm_loss'] for x in gs],15));axes[0,1].set_title('PCM loss (15-update mean)')
gan=[x for x in gs if x['gan']]
axes[0,2].plot([x['g'] for x in gan],smooth([x['gan_loss'] for x in gan]));axes[0,2].set_title('Generator GAN loss (active updates only)')
for key in ('D_real','D_fake'):
 axes[1,0].plot([x['fake_update']/5 for x in fs],smooth([x[key] for x in fs],40),label=key)
axes[1,0].set_title('Discriminator logits (40 D updates mean)');axes[1,0].legend()
axes[1,1].plot([x['g'] for x in gs],[x['grad_norm'] for x in gs],alpha=.7);axes[1,1].set_title('Student gradient norm before clipping');axes[1,1].set_yscale('log')
axes[1,2].plot([x['g'] for x in gs],[x['cycle_seconds'] for x in gs]);axes[1,2].set_title('Seconds / full cycle (5 fake/D + 1 G)');axes[1,2].axvline(2.5,ls='--',color='gray',label='micro4 -> micro16');axes[1,2].legend()
for ax in axes.flat:ax.set_xlabel('Student update G');ax.grid(alpha=.15)
fig.suptitle('Uniform 4-step SDXL, 10K training set, joint4 | diagnostics, not image-quality benchmarks')
fig.tight_layout();fig.savefig(out/'training_curves.png');fig.savefig(out/'training_curves.pdf');plt.close(fig)
print(json.dumps({'wall_hours_resume':r['guard']['wall_seconds']/3600,'mean_cycle':r['steady_mean_cycle_seconds'],'prompts':[(c['split'],c['original_index'],c['prompt']) for c in cases]},ensure_ascii=False,indent=2))
