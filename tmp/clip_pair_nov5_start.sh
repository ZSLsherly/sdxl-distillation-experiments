set -eu
. /root/sdxl_research/env.sh
out="$SDXL_WORK/semantic_pair_1024_20261105"
mkdir -p "$out"
export PYTHONPATH="$SDXL_ROOT/tmp/bc_coco_winner_20261102/code"
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$out"
test -f results.json && exit 0
if test -s worker.pid && kill -0 "$(cat worker.pid)" 2>/dev/null; then echo RUNNING; exit 0;fi
python - <<'PY'
from pathlib import Path
import os,json,tarfile
r=Path(os.environ['SDXL_ROOT']);base=r/'outputs/bc_winner_late_mix_20261102'
dirs={'A':base/'coco10k_A','A_pcm10':r/'outputs/A_pcm10_g160/A_pcm10','C_mix1':base/'coco10k_C_mix1','PCM_DM4_G160':r/'outputs/c_mix1_pcm_dm4_matched_time/PCM_DM4_G160'}
m=json.loads((dirs['A']/'manifest.json').read_text()); picks=m['visual_indices'][:8]
with tarfile.open('examples.tar','w') as t:
 for arm,d in dirs.items():
  n=json.loads((d/'manifest.json').read_text());assert n['samples']==m['samples'] and n['visual_indices']==m['visual_indices']
  for i in picks:t.add(d/'images'/arm/f'{i:05d}.png',arcname=f'{arm}/{i:05d}.png')
Path('cases.json').write_text(json.dumps([m['samples'][i] for i in picks]))
s=(r/'tmp/clip_four_1024.py').read_text().replace("out=Path(os.environ['SDXL_WORK'])/'semantic_four_1024_20261103'","out=Path.cwd()")
s=s[:s.index("folders={")]+f"folders={{k:Path(v) for k,v in { {k:str(v) for k,v in dirs.items() if k in ('A_pcm10','PCM_DM4_G160')}!r}.items()}}\n"+s[s.index('mb=read('):]
s=s.replace("folders['B']","folders['A_pcm10']")
s=s[:s.index('pairs=[')]+"""baseline=read(out.parent/'semantic_four_1024_20261103/results.json')
assert [(c['index'],c['seed'],c['prompt']) for c in baseline['cases']]==[(i,mb['samples'][i]['seed'],p) for i,p in zip(ids,prompts)]
assert selection['clip_revision']==baseline['revision']
for a in ('A','C_mix1'):
    per[a]=np.array(baseline['per_case'][a]['alignment'])
    metrics[a]=baseline['metrics'][a]
pairs=[('A_pcm10','A'),('PCM_DM4_G160','C_mix1')]
"""+s[s.index('def paired(start)'):]
s=s[:s.index('old=read(old_results)')]+"replication={}\n"+s[s.index("result={'model'"):]
Path('score.py').write_text(s)
PY
test -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)"
nohup python -u score.py >worker.log 2>&1 </dev/null & echo $! >worker.pid
sleep 2; tail -4 worker.log;cat worker.pid

