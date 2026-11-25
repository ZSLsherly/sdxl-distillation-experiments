ROOT=/data5/shenglin.zeng
sed -n '805,820p' "$ROOT/src/SparkDiffusion/sparkdiffusion/networks/wan2pt1.py"
sed -n '925,945p' "$ROOT/src/SparkDiffusion/sparkdiffusion/networks/wan2pt1.py"
tail -5 "$ROOT/logs/train_smoke.log"
