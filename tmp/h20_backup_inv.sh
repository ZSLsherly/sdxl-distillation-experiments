set -eu
cd /root/autodl-tmp
du -sh sdxl_runs/outputs/* h20_C_mix1_formal_20261103 h20_direct_after_coco_20261103 h20_coco_after_g160 2>/dev/null
df -h .
find sdxl_runs/outputs -type l -printf '%p -> %l\n' | head -16
