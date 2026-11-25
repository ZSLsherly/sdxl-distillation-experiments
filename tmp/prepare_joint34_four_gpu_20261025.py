from pathlib import Path
import json,sys,os,hashlib
ROOT=Path('/data1/shenglin.zeng/sdxl_eval_uniform_20261025')
old=ROOT/'outputs/joint34_uniform_10k_2gpu_20261025'
out=ROOT/'outputs/joint34_uniform_10k_4gpu_20261025';out.mkdir(exist_ok=False)
sys.path.insert(0,str(ROOT/'training_code'))
from launch import flags,gpu_check
from batching import BatchPlan
import train
proof=json.loads((ROOT/'joint34_reference_provenance.json').read_text())
for name,h in proof['source_sha256'].items():assert hashlib.sha256((ROOT/'training_code'/name).read_bytes()).hexdigest()==h
for phase,micro,old_micro in [('phase1',2,4),('train',8,16)]:
 cfg=json.loads((old/(phase+'_config.json')).read_text())
 cfg.update(micro_batch=micro,rollout_batch=micro,fake_offload=False,offload_student_optimizer=False)
 sys.argv=['train.py','--root',str(ROOT),'--run',out.name+'/'+phase,*flags(cfg)];args=train.arguments()
 plan=BatchPlan(64,4,micro);ref=BatchPlan(64,2,old_micro)
 assert plan.accumulation==ref.accumulation
 for update in [0,1,159,799]:
  new=sorted(int(p) for rank in range(4) for mi in range(plan.accumulation) for p in plan.positions(update,mi,rank))
  prev=sorted(int(p) for rank in range(2) for mi in range(ref.accumulation) for p in ref.positions(update,mi,rank))
  assert new==prev and len(set(new))==64
 (out/(phase+'_config.json')).write_text(json.dumps(cfg,indent=2))
design=json.loads((old/'design.json').read_text())
design.update(world_size=4,microbatch_schedule={'G1_G2':2,'G3_G160':8},runtime_adjustments={'reference_world':2,'new_world':4,'reference_fake_offload':True,'new_fake_offload':False,'reference_offload_student_optimizer':True,'new_offload_student_optimizer':False,'accumulation_unchanged':[8,2]},hardware_note='Reference: two A100-labelled 40GB GPUs; new: four RTX4090-labelled 48GB GPUs. Same global samples and accumulation; runtime layout differs. No bitwise identity claim.',failed_attempt=str(old),reason_for_runtime_change='Parent user.slice memory.high=100GiB and memory.max=112GiB; avoid CPU offload pressure without modifying server limits.',batch_plan_validation='Four-rank and reference two-rank global sample positions exactly match for updates 0,1,159,799; each update has 64 unique positions.')
(out/'design.json').write_text(json.dumps(design,indent=2))
print('IDLE_GPU_CHECK',gpu_check([2,3,5,6],36))
print('PREPARED',json.dumps(design))
