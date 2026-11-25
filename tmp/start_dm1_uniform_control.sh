set -eu
. /root/sdxl_research/env.sh
o="$SDXL_ROOT/outputs/dm1_mid4_uniform_mix1_g160"; c="$SDXL_ROOT/tmp/dm1_mid4_uniform_mix1_g160/code"
b="$SDXL_ROOT/outputs/c_mix1_first_dm_fake3_g160/model"
mkdir -p "$o" "$c"; exec 9>"$o/deploy.lock";flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null;then echo ALREADY_QUEUED;exit 0;fi
cd "$c"
cp "$b/source/"*.py .
cp "$SDXL_ROOT/tmp/c_mix1_first_dm_fake3_g160/code/"{run_quota_bc,launch}.py .
cp "$SDXL_ROOT/tmp/c_mix1_pcm_dm4_matched_time/code/queue_dm_pair_coco.py" .
tar xzf "$SDXL_ROOT/tmp/dm1_uniform_reviewed.tgz"
python - <<'PY'
from pathlib import Path
import hashlib,json,os
root=Path(os.environ['SDXL_ROOT'])
hashes=json.loads((root/'outputs/c_mix1_first_dm_fake3_g160/model/source_sha256.json').read_text())
for n,h in hashes.items():
 if n.endswith('.py'):assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
p=Path('train.py');s=p.read_text();old="    config['turning_sampling']['scope']='student and fake boundary DM'"
assert s.count(old)==1
p.write_text(s.replace(old,"    if turning:\n    "+old))
# Reuse the existing fixed 128-image scoring pipeline.
p=Path('dm_pair_clip128.sh');s=p.read_text()
s=s.replace("('DM234_fake3','DM1_mid4_fake3')","('DM1_mid4_uniform',)").replace("folder=root/'c_mix1_dm_pair_coco10k'/arm","folder=root/'dm1_mid4_uniform_mix1_g160'/arm")
s=s.replace("v=np.array(vals[arm])-vals['C_mix1']", "v=np.array(vals[arm])-np.array(json.loads((root/'c_mix1_dm_pair_coco10k/clip128.json').read_text())['per_case']['DM1_mid4_fake3'])")
s=s.replace('delta_vs_C_mix1','delta_vs_DM1_mid4_fake3').replace("(root/'c_mix1_dm_pair_coco10k/clip128.json').write_text","(root/'dm1_mid4_uniform_mix1_g160/clip128.json').write_text")
s=s.replace("v=np.array(vals[arm])-np.array(json.loads((root/'c_mix1_dm_pair_coco10k/clip128.json').read_text())['per_case']['DM1_mid4_fake3'])",
"""parent=json.loads((root/'c_mix1_dm_pair_coco10k/clip128.json').read_text())
  assert parent['n']==128 and parent['cases']==cases
  v=np.array(vals[arm])-np.array(parent['per_case']['DM1_mid4_fake3'])""")
assert "parent['cases']==cases" in s
Path('clip128.sh').write_text(s)
p=Path('launch.py')
s=p.read_text()
assert s.count("text=True)")==2
p.write_text(s.replace("text=True)","text=True,timeout=15)"))
PY
export PYTHONPATH="$c:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 TORCHDYNAMO_DISABLE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CUDA_VISIBLE_DEVICES='' python audit_dm1_cpu.py
offset=$(wc -c < "$o/queue.log" 2>/dev/null || echo 0)
offset=$((offset+1))
nohup python -u run_dm1_uniform_control.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid"
for i in $(seq 1 30);do tail -c +"$offset" "$o/queue.log" | grep -q VERIFIED && break;kill -0 "$(cat "$o/queue.pid")" || break;sleep 1;done
tail -c +"$offset" "$o/queue.log";kill -0 "$(cat "$o/queue.pid")";tail -c +"$offset" "$o/queue.log" | grep -q VERIFIED;cat "$o/queue.pid"

