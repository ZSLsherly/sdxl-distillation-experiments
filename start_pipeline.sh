set -eu
ROOT=/data5/shenglin.zeng
nohup bash "$ROOT/wait_and_run.sh" > "$ROOT/logs/smoke_pipeline.log" 2>&1 < /dev/null &
printf 'PIPELINE_PID=%s\n' "$!"
printf '\nSOURCE REVISION\n'
git -C "$ROOT/src/SparkDiffusion" rev-parse HEAD
printf '\nOWN PYTHON PATHS\n'
source "$ROOT/env.sh"
python -c 'import sys,ssl; print(sys.executable); print(sys.base_prefix); print(ssl.OPENSSL_VERSION)'
