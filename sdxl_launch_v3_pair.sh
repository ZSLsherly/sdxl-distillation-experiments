set -euo pipefail
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
test "$(cat logs/gan_v3_preflight.exit)" = 0
test -f outputs/gan_stage_v3/preflight/status.json
python - <<'PY'
import json
from pathlib import Path
p=Path('outputs/gan_stage_v3/preflight/status.json');s=json.loads(p.read_text())
assert s['status']=='preflight_complete' and s['student_updates']==2 and s['fake_updates']==2
assert json.loads(Path('outputs/gan_v3_calibration.json').read_text())['gan_weight']>0
assert not Path('outputs/gan_stage_v3/launch.json').exists(),'This pair is already launched'
PY
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_DEBUG=WARN OMP_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1
runstamp=$(TZ=Asia/Shanghai date +%Y%m%dT%H%M%S)
export RUNSTAMP="$runstamp"
mkdir -p outputs/gan_stage_v3
nohup setsid bash -c 'CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 src/sdxl_gan_train_v3.py --run "gan_stage_v3/joint4_${RUNSTAMP}" --variant joint4 --batch 64 --hours 24 > "logs/gan_v3_joint4_${RUNSTAMP}.log" 2>&1; echo $? > "logs/gan_v3_joint4_${RUNSTAMP}.exit"' < /dev/null > "logs/gan_v3_joint4_${runstamp}_launcher.log" 2>&1 &
p_joint=$!
nohup setsid bash -c 'CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 src/sdxl_gan_train_v3.py --run "gan_stage_v3/gan4_${RUNSTAMP}" --variant gan4 --batch 64 --hours 24 > "logs/gan_v3_gan4_${RUNSTAMP}.log" 2>&1; echo $? > "logs/gan_v3_gan4_${RUNSTAMP}.exit"' < /dev/null > "logs/gan_v3_gan4_${runstamp}_launcher.log" 2>&1 &
p_gan=$!
python - "$runstamp" "$p_joint" "$p_gan" <<'PY'
import json,sys
from pathlib import Path
s,j,g=sys.argv[1:]
record={'timestamp_Asia_Shanghai':s,'joint4':{'gpus':[0,1],'launcher_pid':int(j),'run':'gan_stage_v3/joint4_'+s},'gan4':{'gpus':[2,3],'launcher_pid':int(g),'run':'gan_stage_v3/gan4_'+s},'effective_batch_each':64,'eval_student_samples':128,'weight_snapshot_seconds':1800,'hours_limit':24}
Path('outputs/gan_stage_v3/launch.json').write_text(json.dumps(record,indent=2));print(json.dumps(record),flush=True)
PY
