ROOT=/data5/shenglin.zeng
date -Is
printf '\nDOWNLOAD\n'; tail -3 "$ROOT/logs/segmented_download.log"
du -h "$ROOT/models/Wan2.1-T2V-1.3B/models_t5_umt5-xxl-enc-bf16.pth.segmented" 2>/dev/null || true
printf '\nPIPELINE\n'; tail -6 "$ROOT/logs/smoke_pipeline.log"
for file in prepare_embeddings prepare_latents train_smoke; do
 if [ -f "$ROOT/logs/$file.log" ]; then printf '\n%s\n' "$file"; tail -8 "$ROOT/logs/$file.log"; fi
done
if [ -f "$ROOT/logs/smoke_pipeline.exit" ]; then printf '\nEXIT\n'; cat "$ROOT/logs/smoke_pipeline.exit"; fi
