set -eu
r=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
d="$r/outputs/semantic_references_1024"
mkdir "$d";cd "$d";tar -xzf /data1/shenglin.zeng/semantic_refs.tgz
export PYTHONPATH="$r/code" LD_LIBRARY_PATH="$r/runtime/lib:${LD_LIBRARY_PATH:-}" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nohup "$r/env/bin/python" -u run_references.py </dev/null >run.log 2>&1 & echo QUEUE_PID=$!
