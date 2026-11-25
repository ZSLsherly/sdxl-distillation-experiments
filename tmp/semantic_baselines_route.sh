set -eu
r=/data5/shenglin.zeng/sdxl_crossdistill
cd "$r"
ls -lh models/baselines/dmd2/unet.safetensors
stat -c "%U" "$r"
for a in sdxl dmd2;do find "outputs/coco10k_g160_20261020/images/$a" -name '*.png' | wc -l;done
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ConnectTimeout=3 shenglin.zeng@172.18.132.41 true && echo 'existing passwordless route usable' || true
