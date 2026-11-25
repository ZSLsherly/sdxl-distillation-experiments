set -eu
root=/data1/shenglin.zeng/sdxl_eval_uniform_20261025
du -sh "$root/models/sdxl-base-1.0" "$root/datasets/parallel_10k_seed20261018" "$root/env" "$root/runtime" "$root/transfers" 2>/dev/null || true
ls -lh "$root/transfers" 2>/dev/null || true
readlink -f "$root/env/bin/python"
test -f "$root/transfers/environment.tar.gz" && echo ENV_ARCHIVE_AVAILABLE || true
test -f "$root/runtime/bin/python" && echo RUNTIME_AVAILABLE || true
