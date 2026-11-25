set -eu
date -Is
for root in /data1/shenglin.zeng/sdxl_eval_uniform_20261025 /data1/shenglin.zeng/teacher_turning_20261026 /data5/shenglin.zeng/sdxl_crossdistill;do
 if test -d "$root";then
  printf '\nROOT %s\n' "$root"
  find "$root" -maxdepth 6 -type f \( -name 'student_rank0.pt' -o -name 'shapes_rank0.json' -o -iname '*lpips*' -o -iname '*dino*' -o -name 'config.json' \) | head -65
 fi
done
