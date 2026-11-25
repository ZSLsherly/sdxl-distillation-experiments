set -e
R=/data1/shenglin.zeng/dmd2_matched_10k_20261119
export CUDA_VISIBLE_DEVICES=GPU-3a1e6a0e-9fab-18ef-c543-30066e01e383
export OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1
"$R/env/bin/python" -u "$R/code/check_native_sampler.py" "$R"
stat -c '%y %s %n' "$R/runs/formal_g160_restart_20261120/complete.json" "$R/runs/formal_g160_restart_20261120/student.pt"
