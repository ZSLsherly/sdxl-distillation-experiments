cd /data5/shenglin.zeng/sdxl_crossdistill
for p in outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16 outputs/gan_stage_joint34_4gpu/joint34_20261019T195324_4gpu_mb16 outputs/gan_stage_parallel/joint4_20261017T232317_mb16; do
  echo "$p"
  cat "$p/status.json"
done
