set -eu
p=/data5/shenglin.zeng
find "$p/checkpoints/h20_gan_20261104" -maxdepth 7 -name '*.npz' | head -4
ls -lh "$p/sdxl_crossdistill/models/metrics/clean_fid"
find "$p/sdxl_crossdistill" -maxdepth 4 -name coco10k.zip -ls
