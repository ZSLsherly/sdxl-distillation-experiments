set -eu
root=/data5/shenglin.zeng/sdxl_crossdistill/exports
for p in "$root/turning_20261026" "$root/coco_uniform_g160_20261025";do
 if test -d "$p";then ls -lh "$p" | head -20;fi
done
