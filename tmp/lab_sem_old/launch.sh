set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$d/outputs/semantic_joint4_uniform_1024"
export PATH="$d/env/bin:$PATH" PYTHONPATH="$d/code" LD_LIBRARY_PATH="$d/runtime/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
mkdir -p images
exec 9>launch.lock
flock -n 9
test ! -f results.json
bash fetch.sh >fetch.log 2>&1 & fetch_pid=$!
python -u gen.py check
mapfile -t free < <(python - <<'PY'
import subprocess
def q(a,b):return subprocess.check_output(['nvidia-smi','--query-'+a+'='+b,'--format=csv,noheader,nounits'],text=True).splitlines()
busy={r.split(',')[0].strip() for r in q('compute-apps','gpu_uuid,pid')}
for r in q('gpu','index,uuid,memory.free,utilization.gpu'):
 i,u,m,v=map(str.strip,r.split(','))
 if u not in busy and int(v)==0 and int(m)>42000:print(u)
PY
)
n=${#free[@]}; if (( n>4 ));then n=4;fi; test "$n" -gt 0
pids=()
for ((i=0;i<n;i++));do echo "START worker $i ${free[$i]}"; CUDA_VISIBLE_DEVICES="${free[$i]}" python -u gen.py "$i" "$n" >"gen$i.log" 2>&1 & pids+=($!);done
rc=0;for pid in "${pids[@]}";do wait "$pid" || rc=1;done; test "$rc" -eq 0
wait "$fetch_pid"
CUDA_VISIBLE_DEVICES='' python -u score.py >score.log 2>&1
