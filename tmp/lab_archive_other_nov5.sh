set -eu
cd /data5/shenglin.zeng
find . -type f \( -name prepared.pt -o -name 'experiments.tar' \) -printf '%p %s\n' |head -80
cd checkpoints
find a800_abc_g160_20261102 a800_20261103 -type f -name student_rank0.pt -size +4G -exec sha256sum {} +

