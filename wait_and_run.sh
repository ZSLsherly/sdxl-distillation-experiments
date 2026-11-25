#!/usr/bin/env bash
set -euo pipefail
ROOT=/data5/shenglin.zeng
for attempt in $(seq 1 180); do
 if grep -q '^ALL_MODEL_FILES_COMPLETE$' "$ROOT/logs/segmented_download.log"; then
  exec bash "$ROOT/run_smoke.sh"
 fi
 if ! kill -0 2977635 2>/dev/null; then printf 'Model download stopped before verification\n'; exit 1; fi
 sleep 10
done
printf 'Download wait exceeded 30 minutes\n'
exit 1
