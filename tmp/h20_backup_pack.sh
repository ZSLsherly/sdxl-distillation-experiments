set -eu
cd /root/autodl-tmp
b=h20_backup_20261104
mkdir -p "$b"
w=sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160/weights/20261104T112222+0800_g000160_end/student_rank0.pt
real=sdxl_runs/outputs/h20_C_mix1_terminal_GAN_G160_coco10k/features/real
tar -cf "$b/experiments.tar.partial" --exclude="$w" --exclude="$real" sdxl_runs/outputs h20_C_mix1_formal_20261103 h20_direct_after_coco_20261103 h20_coco_after_g160
tar -rhf "$b/experiments.tar.partial" "$real"
mv "$b/experiments.tar.partial" "$b/experiments.tar"
sha256sum "$b/experiments.tar" "$w" > "$b/SHA256SUMS"
cat "$b/SHA256SUMS"
