set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
trap 'rc=$?; echo "$rc" > logs/gan_stage_pilot.exit' EXIT
python -m py_compile src/sdxl_gan_train.py src/sdxl_pixel_gan.py
test -f outputs/gan_stage_pilot/calibration/summary.json
echo 'PREFLIGHT_START joint34; 3 student updates, 15 fake+D updates; all four GPUs'
torchrun --standalone --nproc_per_node=4 src/sdxl_gan_train.py --run gan_stage_pilot/preflight --variant joint34 --student-updates 3 --preflight > logs/gan_preflight.log 2>&1
python -c 'import json; from pathlib import Path; p=Path("outputs/gan_stage_pilot/preflight"); s=json.loads((p/"summary.json").read_text()); assert s["status"]=="preflight_complete"; rows=[json.loads(l) for l in (p/"rank0.jsonl").read_text().splitlines()]; g=[r for r in rows if r["phase"]=="student"]; assert [r["generator_stage"] for r in g]==[2,3,4]; assert all(r["gan_latent_gradient_rms"]>0 for r in g if r["gan_active"]); assert any(r.get("r1",0)>0 for r in rows); assert all((p/f"summary_rank{i}.json").exists() for i in range(4)); print("PREFLIGHT_PASSED",s["peak_allocated_GiB"])'
for variant in dmd gan4 joint4 joint34; do
  echo "TRIAL_START $variant; 24 student updates, 120 fake+D updates; all four GPUs"
  torchrun --standalone --nproc_per_node=4 src/sdxl_gan_train.py --run "gan_stage_pilot/$variant" --variant "$variant" --student-updates 24 > "logs/gan_${variant}.log" 2>&1
  python -c "import json; from pathlib import Path; p=Path('outputs/gan_stage_pilot/$variant'); s=json.loads((p/'summary.json').read_text()); assert s['status']=='training_complete' and s['student_updates']==24; assert all((p/f'student_rank{i}.pt').is_file() for i in range(4)); print('TRIAL_PASSED','$variant')"
done
echo ALL_FOUR_TRIALS_COMPLETE
