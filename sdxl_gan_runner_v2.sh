set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
trap 'rc=$?; echo "$rc" > logs/gan_v2_series.exit' EXIT
test "$(cat logs/gan_v2_preflight.exit)" = 0
python src/sdxl_gan_audit_v2.py outputs/gan_stage_pilot_v2/preflight --preflight
for variant in dmd gan4 joint4 joint34_matched; do
  test ! -f outputs/gan_stage_pilot_v2/STOP_REQUEST
  echo "TRIAL_START $variant"
  torchrun --standalone --nproc_per_node=4 src/sdxl_gan_train_v2.py --run "gan_stage_pilot_v2/$variant" --variant "$variant" --student-updates 24 > "logs/gan_v2_${variant}.log" 2>&1
  python src/sdxl_gan_audit_v2.py "outputs/gan_stage_pilot_v2/$variant"
done
test ! -f outputs/gan_stage_pilot_v2/STOP_REQUEST
echo ALL_V2_TRAINING_COMPLETE
python src/sdxl_gan_evaluate_v2.py > logs/gan_v2_assessment.log 2>&1
python src/sdxl_gan_render_report_v2.py >> logs/gan_v2_assessment.log 2>&1
echo V2_REPORT_COMPLETE
