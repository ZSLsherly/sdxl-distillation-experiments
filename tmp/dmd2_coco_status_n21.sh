R=/data1/shenglin.zeng/dmd2_matched_10k_20261119/coco10k_official_20261121
date
for f in queue prepare probe metrics; do [ ! -f "$R/logs/$f.log" ] || { echo "$f"; tail -n 7 "$R/logs/$f.log"; }; done
for f in "$R"/status_*.json "$R"/complete.json "$R"/probe_passed.json; do [ ! -f "$f" ] || cat "$f"; done
