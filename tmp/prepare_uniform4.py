import difflib, hashlib, json, shutil, tarfile
from pathlib import Path

root = Path(__file__).resolve().parent.parent
dest = root / 'sdxl_uniform4_20261022'
dest.mkdir(exist_ok=False)
files = ('train.py','batching.py','pixel.py','objectives.py','training_state.py','test_batching.py')
for name in files:
    shutil.copyfile(root/'sdxl_parallel'/name, dest/name)

def replace(name, old, new, count=1):
    p=dest/name
    s=p.read_text(encoding='utf-8')
    assert s.count(old)==count, (name,old,s.count(old))
    p.write_text(s.replace(old,new),encoding='utf-8')

replace('train.py','knots = [999, 919, 499, 199, -1]','knots = [999, 749, 499, 249, -1]')
replace('train.py','seed=args.seed)','seed=args.seed, boundary=knots[1])',2)
replace('train.py','th - 20, 919)','th - 20, knots[1])')
replace('train.py','th, 919)','th, knots[1])')
replace('train.py','"student_cfg": 1, "train_count": data["train_count"],','"student_cfg": 1, "train_count": data["train_count"],\n              "student_dmd_t_range": [20, knots[1]], "pcm_t_range": [knots[1] + 21, 999],')
replace('batching.py','shape=(4, 128, 128)):','shape=(4, 128, 128), boundary=919):')
replace('batching.py','else (20, 920)','else (20, boundary + 1)')
replace('batching.py','torch.randint(940, 1000','torch.randint(boundary + 21, 1000')
replace('training_state.py','keys = ("variant",','keys = ("schedule", "variant",')
diff=''.join(line for name in files for line in difflib.unified_diff(
    (root/'sdxl_parallel'/name).read_text(encoding='utf-8').splitlines(True),
    (dest/name).read_text(encoding='utf-8').splitlines(True),fromfile='baseline/'+name,tofile='uniform4/'+name))
(dest/'changes.diff').write_text(diff,encoding='utf-8')
protocol={
 'status':'prepared; GPU short test NOT run; target GPU4/5 require reset and GPU6 occupied at preflight',
 'server':'172.18.132.41','physical_gpus':[4,5,6,7],
 'baseline':'gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16',
 'initialization':'original SDXL; independent student/fake/discriminator; no resume',
 'schedule':[999,749,499,249,-1], 'sampler':'original deterministic VP/DDIM jump, unchanged',
 'student_dmd_t_range':[20,749], 'pcm_t_range':[770,999], 'pcm_boundary':749,
 'fake_t_range':[0,999], 'pcm_teacher_substeps':[10,10],
 'student_stage_cycle':[2,3,4], 'gan_stages':[4], 'pcm_weight':10,
 'gan_weight':0.005274229682981968,'student_lr':1e-6,'fake_lr':1e-6,'d_lr':1e-4,
 'teacher_cfg':8,'student_cfg':1,'seed':20261017,
 'effective_batch':64,'world_size':4,'micro_batch':16,'rollout_batch':16,'vae_batch':4,
 'fake_updates_per_student':5,'max_student_updates':160,'save_every_updates':40,'eval_every_updates':2,
 'data':'same datasets/parallel_10k_seed20261018 prepared.pt and manifest.json',
 'final_checkpoint':'full FP32 student/fake/D, optimizer states, counters, RNG',
 'short_test':'fresh separate 4-student-update run with full 5:1 fake/D schedule, includes R1, GAN stage4, periodic evaluation and final save; then fresh G160 only after successful review',
 'interpretation':'uniform four-stage partition; TM boundary and its associated sampling ranges move together; not a sampler-only ablation',
 'source_sha256':{name:hashlib.sha256((dest/name).read_bytes()).hexdigest() for name in files},
}
(dest/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
print(diff)
print(dest)
