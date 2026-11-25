set -eu
d=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
ls "$d" "$d/outputs/coco10k" "$d/checkpoints/weights/g160"
ls -l "$d/env/bin/python" "$d/code"
find /data1/shenglin.zeng -type f \( -name 'ViT-B-32.pt' -o -name 'pytorch_model.bin' -o -name 'revision.txt' \) | head -30
find "$d/outputs/coco10k/images/joint4_uniform" -name '*.png' | wc -l
df -h "$d"
