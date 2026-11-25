set -eu
. /root/sdxl_research/env.sh
c="$SDXL_ROOT/tmp/A_pcm10_mix1_g160/code";o="$SDXL_ROOT/outputs/A_pcm10_mix1_g160"
mkdir -p "$c" "$o";cd "$c";exec 9>"$o/deploy.lock";flock -n 9
if test -s "$o/queue.pid" && kill -0 "$(cat "$o/queue.pid")" 2>/dev/null;then echo ALREADY_ACTIVE;exit 0;fi
test ! -f "$o/train.log"
cp "$SDXL_ROOT/outputs/A_pcm10_g160/model/source/"*.py .
cp "$SDXL_ROOT/tmp/A_pcm10_g160/code/launch.py" .
cp "$SDXL_ROOT/outputs/bc_winner_late_mix_20261102/late_mix/source/fake_tracking.py" .
cp "$SDXL_ROOT/tmp/run_A10_mix1.py" .
export PYTHONPATH="$c:$SDXL_ROOT/tmp/dm1_mid4_uniform_mix1_g160/code:$SDXL_ROOT/tmp/bc_coco_winner_20261102/code:$SDXL_ROOT/code/sdxl_research" CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TORCHDYNAMO_DISABLE=1
python - <<'PY'
from pathlib import Path
import os,json,hashlib,torch
root=Path(os.environ['SDXL_ROOT'])
for n,h in json.loads((root/'outputs/A_pcm10_g160/model/source_sha256.json').read_text()).items():
 assert hashlib.sha256(Path(n).read_bytes()).hexdigest()==h,n
p=Path('train.py');s=p.read_text()
a='    args = parser.parse_args()'
assert s.count(a)==1
s=s.replace(a,'    parser.add_argument("--fake-student-mix", type=float, default=0)\n'+a+'\n    if args.fake_student_mix not in (0,.01): parser.error("Expected 0 or 1% mixing")')
a='        gcount += 1\n'
assert s.count(a)==1
s=s.replace(a,a+'''        if args.fake_student_mix:
            from fake_tracking import mix_shared_parameters
            mix_shared_parameters(fake,student,args.fake_student_mix)
            record(dict(phase='fake_student_mix',g=gcount,rate=args.fake_student_mix))
''')
a='"stage_policy.py", "gan_losses.py"'
assert s.count(a)==1
s=s.replace(a,'"fake_tracking.py", '+a)
compile(s,'train.py','exec');p.write_text(s)
from fake_tracking import mix_shared_parameters
st=torch.nn.Linear(3,2);fk=torch.nn.Linear(3,2);fk.class_embedding=torch.nn.Embedding(3,2)
before={k:v.clone() for k,v in fk.named_parameters()};original={k:v.clone() for k,v in st.named_parameters()}
mix_shared_parameters(fk,st,.01)
for k,v in st.named_parameters():
 torch.testing.assert_close(v,original[k])
 torch.testing.assert_close(dict(fk.named_parameters())[k],.99*before[k]+.01*v)
torch.testing.assert_close(fk.class_embedding.weight,before['class_embedding.weight'])
print('CPU_MIX_OK',flush=True)
PY
nohup python -u run_A10_mix1.py 9>&- >>"$o/queue.log" 2>&1 </dev/null &
echo $! >"$o/queue.pid";sleep 5;tail -6 "$o/queue.log";kill -0 "$(cat "$o/queue.pid")";cat "$o/queue.pid"

