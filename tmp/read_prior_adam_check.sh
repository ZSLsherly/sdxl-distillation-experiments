source /data5/shenglin.zeng/env.sh
cd /data5/shenglin.zeng/sdxl_crossdistill
if [ -f outputs/gan_stage_joint34/cpu_adam_check.log ]; then tail -n 8 outputs/gan_stage_joint34/cpu_adam_check.log; fi
