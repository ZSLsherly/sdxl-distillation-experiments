from pathlib import Path
import sys,json,textwrap,os
os.environ['MPLCONFIGDIR']=str(Path('output/sdxl-crossdistill/.matplotlib').resolve())
sys.path.insert(0,str(Path('.plot_runtime_pkgs').resolve()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

root=Path('output/sdxl-crossdistill/hparam_pilot/teacher_stages')
plt.rcParams.update({'font.family':'DejaVu Sans','figure.facecolor':'#fafafa'})
for rank in range(4):
    data=json.loads((root/f'rank{rank}.json').read_text());record=data['records'][0]
    fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
    for ax,row in zip(axes,record['stages']):
        path=root/Path(row['predicted_clean_image']).name
        ax.imshow(Image.open(path));ax.axis('off')
        ax.set_title(f"t = {row['t']}  |  log SNR = {row['log_snr']:.2f}",fontsize=13)
    fig.suptitle(f"SDXL teacher clean estimates | Same seed: 10 | CFG: 8\n{record['prompt'].strip()}",fontsize=15)
    fig.savefig(root/f'prompt{rank}_four_stages.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(4,4,figsize=(14,15),layout='constrained')
    for i,rec in enumerate(data['records']):
        for j,row in enumerate(rec['stages']):
            ax=axes[i,j];ax.imshow(Image.open(root/Path(row['predicted_clean_image']).name));ax.set_xticks([]);ax.set_yticks([])
            if i==0:ax.set_title(f"t = {row['t']} | log SNR = {row['log_snr']:.2f}",fontsize=12)
            if j==0:ax.set_ylabel(f"Seed {rec['seed']}",fontsize=13)
    fig.suptitle(f"Teacher trajectory: stage x seed\n{record['prompt'].strip()}",fontsize=16)
    fig.savefig(root/f'prompt{rank}_stages_and_seeds.png',dpi=120);plt.close(fig)
fig,axes=plt.subplots(1,4,figsize=(16,4.7),layout='constrained')
for rank,ax in enumerate(axes):
    data=json.loads((root/f'rank{rank}.json').read_text());rec=data['records'][0]
    ax.imshow(Image.open(root/Path(rec['final_image']).name));ax.axis('off')
    ax.set_title('\n'.join(textwrap.wrap(rec['prompt'].strip(),35)),fontsize=10)
fig.suptitle('Full teacher DDIM trajectory endpoints | Seed 10 | CFG 8',fontsize=15)
fig.savefig(root/'teacher_final_references.png',dpi=130);plt.close(fig)
print('Teacher scientific figures saved:',root.resolve())
