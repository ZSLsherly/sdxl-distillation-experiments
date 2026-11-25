set -eu
r=/data5/shenglin.zeng
find "$r/checkpoints" -maxdepth 5 -type f \( -name 'status.json' -o -name 'complete.json' -o -name 'student_rank0.pt' -o -name 'manifest.json' \) | head -60
find "$r" -maxdepth 6 -name manifest.json | grep -E 'coco|eval|export' | head -20
find "$r" -maxdepth 5 -type d \( -name 'clean_fid' -o -name 'real' -o -name 'bc_texture_inputs' \)
