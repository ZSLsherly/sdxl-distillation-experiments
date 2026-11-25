set -eu
r=/data1/shenglin.zeng/sdxl_eval_uniform_20261025;cd "$r/outputs/semantic_references_1024"
grep -F "][:1]" run_references.py >/dev/null
grep -F "GPU-c374184a-ab5a-681a-ed48-025bed2da39b" run_references.py >/dev/null
grep -F "GPU-baca962f-0eff-b198-1677-8c28d525ecc1" generate.py >/dev/null
export PYTHONPATH="$r/code" LD_LIBRARY_PATH="$r/runtime/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup "$r/env/bin/python" -u run_references.py >> resume.log 2>&1 </dev/null & echo PID=$!
sleep 2;tail -2 resume.log
