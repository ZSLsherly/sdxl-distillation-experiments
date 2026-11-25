set -euo pipefail
mkdir -p /data1/shenglin.zeng/sdxl_eval_uniform_20261025
export TMPDIR=/data1/shenglin.zeng/sdxl_eval_uniform_20261025/tmp
mkdir -p "$TMPDIR"
nice -n 19 /data1/shenglin.zeng/miniconda3/bin/conda create -y --prefix /data1/shenglin.zeng/sdxl_eval_uniform_20261025/runtime --override-channels -c https://conda.anaconda.org/conda-forge python=3.12
