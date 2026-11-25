set -eu
date -Iseconds
for n in A_clean_global A_terminal_only; do
 r=/data3/shenglin.zeng/new_models_eval_20261114/results/$n
 echo "$n"
 cat "$r/all_complete.json"
 cat "$r/scores.json"
done
