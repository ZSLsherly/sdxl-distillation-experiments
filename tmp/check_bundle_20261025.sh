set -e
ls -lh /data5/shenglin.zeng/sdxl_crossdistill/exports/coco_uniform_g160_20261025/
ps -eo pid,etime,pcpu,args | grep -E 'python -|bundle_coco' | grep -v grep
