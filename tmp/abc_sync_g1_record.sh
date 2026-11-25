set -eu
d=/data5/shenglin.zeng/checkpoints/a800_abc_g160_20261102
mv -f -- "$d/EXPERIMENTS.g1.new" "$d/EXPERIMENTS.txt"
sha256sum "$d/EXPERIMENTS.txt"
