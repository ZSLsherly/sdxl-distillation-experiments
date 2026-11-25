set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/c_mix1_dm234_20261103/code";o="$SDXL_ROOT/outputs/c_mix1_dm234_20261103"
mkdir -p "$c" "$o"
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null; then echo ALREADY_ACTIVE;exit 0;fi
test ! -e "$o/model" && test ! -e "$o/train.log"
src="$SDXL_ROOT/tmp/flash_gan_g160_20261102/code";cp "$src"/*.py "$c/";cp "$SDXL_ROOT/tmp/run_c_mix1_dm234.py" "$c/"
# The original no-GAN runner includes strict CPU checkpoint validation at export.
cd "$c"
export PYTHONPATH="$c" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python - <<'PY'
import ast,json,os,sys
from pathlib import Path
from launch import flags
from train import arguments
root=Path(os.environ['SDXL_ROOT']);prev=root/'outputs/bc_winner_late_mix_20261102';out=root/'outputs/c_mix1_dm234_20261103'
cfg=json.loads((prev/'effective_config.json').read_text());assert json.loads((prev/'selection.json').read_text())['winner']=='C'
assert cfg['fake_student_mix']==.01 and cfg['fake_mix_start_g']==1 and cfg['variant']=='dmd_only'
cfg['both_middle']=True
sys.argv=['train','--run','verify',*flags(cfg)];a=arguments()
assert a.both_middle and a.pair_fake_stages and a.fake_updates==2 and a.batch==64 and a.micro_batch==16 and a.max_student_updates==160 and a.fake_track_pilot and a.gan_weight==0
tr=Path('train.py');s=tr.read_text()
if "write_json(dest/'cpu_validation.json'" not in s:
 old=(root/'tmp/bc_coco_winner_20261102/code/train.py').read_text()
 start=old.index("            if reason=='end' and args.fake_track_pilot")
 end=old.index('            write_json(dest / "complete.json"',start)
 mark='            write_json(dest / "complete.json"'
 assert s.count(mark)==1
 s=s.replace(mark,old[start:end]+mark);compile(s,'train.py','exec');tr.write_text(s)
assert 'verify_tracking_probes' in s and "ids=manifest['holdout_original_indices'][:8]" in s
(out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
print('VALIDATED C+1%; only changed both_middle',flush=True)
PY
nohup python -u run_c_mix1_dm234.py >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
sleep 3
kill -0 "$(cat "$o/queue.pid")"
cat "$o/queue.log"
df -h "$o" | tail -1
