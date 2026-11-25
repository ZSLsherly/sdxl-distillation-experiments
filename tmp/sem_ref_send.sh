set -eu
cd /data5/shenglin.zeng/sdxl_crossdistill/tmp/ref_sem_xfer_20261104
nohup /data5/shenglin.zeng/envs/crossdistill/bin/python -u xfer.py send 44853 >transfer.log 2>&1 </dev/null & echo SEND_PID=$!
sleep 1;tail -3 transfer.log
