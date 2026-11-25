R=/data1/shenglin.zeng/dmd2_official_recipe_20261121
date
find "$R/runs/smoke_micro4/eval" -maxdepth 2 -name '*.png' 2>/dev/null | wc -l
for f in "$R/runs/smoke_micro4/precision_rank0.json" "$R/runs/smoke_micro4/status.json"; do
  if [ -f "$f" ]; then echo "$f"; cat "$f"; fi
done
tail -n 2 "$R/runs/smoke_micro4/rank0.jsonl" 2>/dev/null || true
