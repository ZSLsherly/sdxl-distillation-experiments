set -eu
source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
python -m zipfile -e coco10k_code.zip .
echo running > outputs/coco10k_g160_20261020/verify.exit
nohup bash -c '
  out=outputs/coco10k_g160_20261020
  while [ ! -f "$out/prepare.exit" ]; do sleep 10; done
  if [ "$(cat "$out/prepare.exit")" != 0 ]; then echo PREPARE_FAILED > "$out/verify.exit"; exit 1; fi
  CUDA_VISIBLE_DEVICES=3 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 python -u sdxl_coco10k/validate.py > "$out/verify.log" 2>&1
  result=$?
  echo "$result" > "$out/verify.exit"
' </dev/null >/dev/null 2>&1 &
echo "VALIDATE_WAITER_PID $!"
