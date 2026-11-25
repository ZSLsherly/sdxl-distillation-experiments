set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/c_mix1_dm234_fake3_20261103/code";o="$SDXL_ROOT/outputs/c_mix1_dm234_fake3_20261103"
mkdir -p "$c" "$o"
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null;then echo ALREADY_ACTIVE;exit 0;fi
test ! -e "$o/train.log"
cp "$SDXL_ROOT/tmp/c_mix1_dm234_20261103/code/"*.py "$c/"
tar -xzf "$SDXL_ROOT/tmp/dm234_fake3_patch.tar.gz" -C "$c"
cd "$c";export PYTHONPATH="$c" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python patch_dm234_fake3.py .
python - <<'PY'
from pathlib import Path
import json,os,sys
from launch import flags
from train import arguments
from boundary import critic_stage
from run_dm234_fake3 import out,root
f=Path('run_quota_bc.py');s=f.read_text()
for old,new in [("('complete',160,320,64)","('complete',160,480,64)"),('len(fs)==320','len(fs)==480'),('fs[2*i:2*i+2]]==[g[\'stage\'],4]','fs[3*i:3*i+3]]==[2,3,4]')]:
 assert s.count(old)==1
 s=s.replace(old,new)
compile(s,str(f),'exec');f.write_text(s)
cfg=json.loads((root/'outputs/c_mix1_dm234_20261103/effective_config.json').read_text())
cfg['fake_updates']=3
sys.argv=['train','--run','validate',*flags(cfg)];a=arguments()
assert a.both_middle and a.pair_fake_stages and a.fake_student_mix==.01 and a.fake_mix_start_g==1 and a.variant=='dmd_only'
assert [critic_stage(i,True,a.seed,a.fake_updates) for i in range(480)]==[2,3,4]*160
assert a.batch==64 and a.max_student_updates==160
(out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
print('VERIFIED: 160 G cycles, 480 separate fake steps ordered 2/3/4')
PY
nohup python -u run_dm234_fake3.py >>"$o/queue.log" 2>&1 </dev/null &
echo $! > "$o/queue.pid";sleep 4
kill -0 "$(cat "$o/queue.pid")";cat "$o/queue.log"
