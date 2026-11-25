set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python - <<'PY'
from pathlib import Path
import os,sys,json,torch
r=Path(os.environ['SDXL_ROOT']);s=r/'outputs/turning_BC_paired_quota_20261102/uniform_control/source'
sys.path.insert(0,str(s));from batching import random_inputs
seed=20261017
a=random_inputs([0],'student','cpu',seed=seed+3000,shape=(4,128,128),pcm_boundary=749,time_range=(259,979))
b=random_inputs([1000],'student','cpu',seed=seed+2000,shape=(4,128,128),pcm_boundary=749,time_range=(509,979))
print('CPU_ACTUAL_SHAPE_REPLAY',dict(first_stage=3,first_position=0,second_stage=2,second_position=1000,
 same_generator_seed=True,t_first=a['t'].tolist(),t_second=b['t'].tolist(),epsilon_exact_equal=torch.equal(a['eps'],b['eps'])))
for source in sorted((r/'outputs').glob('*/model/source')):
 cfgp=source.parent/'config.json'
 if not cfgp.exists():continue
 cfg=json.loads(cfgp.read_text());train=(source/'train.py').read_text()
 sampler=(source/'turning_sampling.py').read_text() if (source/'turning_sampling.py').exists() else ''
 prior=cfg.get('turning_sampling') or {}
 profile=source/'turning_distribution.json'
 terminal_prior=json.loads(profile.read_text()).get('terminal_only',False) if profile.exists() else False
 print('DESCENDANT',json.dumps(dict(name=source.parent.parent.name,G=cfg.get('max_student_updates'),
  prior=bool(cfg.get('turning_profile')),terminal_prior_only=terminal_prior,
  terminal_student=bool(cfg.get('terminal_only_dm')) or cfg.get('student_stages')==[4],
  both_middle=cfg.get('both_middle'),fake_updates=cfg.get('fake_updates'),
  additive_prior='random.Random(seed+700000+int(pos))' in sampler,
  tuple_prior='turning:{int(seed)}:{int(pos)}' in sampler,
  additive_stage_noise='seed=args.seed+1000*k' in train),ensure_ascii=False))
PY
