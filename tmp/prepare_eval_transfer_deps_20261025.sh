set -euo pipefail
source /data5/shenglin.zeng/env.sh
export CUDA_VISIBLE_DEVICES=''
export TMPDIR=/data5/shenglin.zeng/tmp
nice -n 19 python -m pip install --disable-pip-version-check --target /data5/shenglin.zeng/tools/eval_transfer_20261025 conda-pack paramiko
