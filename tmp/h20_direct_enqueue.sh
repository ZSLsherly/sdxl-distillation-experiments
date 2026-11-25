set -eu
. /root/sdxl_research/env.sh
d=/root/autodl-tmp/h20_direct_after_coco_20261103
mkdir "$d"
cd "$d"
tar -xzf /root/autodl-tmp/h20_direct_after_coco_20261103.tar.gz
base=/root/autodl-tmp/h20_C_mix1_formal_20261103
coco=/root/autodl-tmp/h20_coco_after_g160
export PYTHONPATH="$coco/vendor:$base:$PWD" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
L=/root/autodl-tmp/h20_gan_mb32/blas126/nvidia/cublas/lib
export LD_PRELOAD="$L/libcublasLt.so.12:$L/libcublas.so.12"
python -m py_compile train.py run.py terminal_gan.py after_coco.py
python run.py --short > short_command.txt
python run.py > formal_command.txt
df -h "$PWD"
nohup python -u after_coco.py </dev/null > queue.log 2>&1 & pid=$!
echo "$pid" > queue.pid
sleep 2
ps -p "$pid" -o pid,etime,stat,args
cat queue.log
