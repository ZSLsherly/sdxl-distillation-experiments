ROOT=/data5/shenglin.zeng
date -Is
find "$ROOT/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke" -maxdepth 5 -type f -name '*.distcp' -printf '%s %p\n' 2>/dev/null | head -20
du -sh "$ROOT/outputs"
tail -6 "$ROOT/logs/train_smoke.log"
test ! -f "$ROOT/logs/smoke_pipeline.exit" || cat "$ROOT/logs/smoke_pipeline.exit"
