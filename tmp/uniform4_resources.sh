cd /data5/shenglin.zeng/sdxl_crossdistill
cat outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16/status.json
cat outputs/gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16/status.json
ps -p 647833,92083,1172438,3574011,1189254,1105413,1117398 -o pid,user,etime,comm
free -h
ls -l /data5/shenglin.zeng/tmp
