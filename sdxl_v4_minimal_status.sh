source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
TZ=Asia/Shanghai date +%H:%M:%S
find outputs/v4_validation/control_20261017T205945/checkpoints -maxdepth 2 -type f -printf '%f %s\n'
tail -n 6 logs/v4_minimal_supervisor.log
