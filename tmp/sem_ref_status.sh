set -eu
cd /data1/shenglin.zeng/sdxl_eval_uniform_20261025/outputs/semantic_references_1024
date -Is;tail -4 run.log;for a in sdxl dmd2;do echo "$a $(find "images/$a" -name '*.png' 2>/dev/null|wc -l) / 896";done
for f in sdxl_?.log dmd2_?.log;do if test -f "$f";then grep -aE 'Error|Traceback|DONE' "$f" |tail -3;fi;done
ls -lh models/baselines/dmd2/* assets.ready *_results.json 2>/dev/null || true
