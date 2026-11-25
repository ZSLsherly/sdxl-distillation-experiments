set -eu
p=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
cd "$p"
sha256sum code/evaluate.py code/evaluation_core.py code/objectives.py models/metrics/clean_fid/inception-2015-12-05.pt
mkdir -p tmp/bc_public_reference_20261102
if ! test -f tmp/bc_public_reference_20261102/reference.tar; then nice -n 15 tar -cf tmp/bc_public_reference_20261102/reference.partial -C "$p" reference_features/real models/metrics/clean_fid/inception-2015-12-05.pt; mv tmp/bc_public_reference_20261102/reference.partial tmp/bc_public_reference_20261102/reference.tar; fi
sha256sum tmp/bc_public_reference_20261102/reference.tar
