set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
CUDA_VISIBLE_DEVICES='' "$R/env/bin/python" "$R/code/test_cpu.py"
nvidia-smi --query-gpu=index,uuid,memory.used,memory.free,utilization.gpu --format=csv,noheader
ps -eo pid,args | grep '[r]un_guard.py' || true
