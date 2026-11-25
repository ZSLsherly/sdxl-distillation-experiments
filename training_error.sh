ROOT=/data5/shenglin.zeng
grep -n -E 'Error|Exception|Traceback|Assertion|out of memory|Starting training|SMOKE_METRIC' "$ROOT/logs/train_smoke.log" | tail -35
tail -130 "$ROOT/logs/train_smoke.log" | head -90
