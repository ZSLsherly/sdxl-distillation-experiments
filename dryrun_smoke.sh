set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python -m scripts.train --config=smoke_config.py --dryrun > "$CROSSDISTILL_ROOT/logs/config_dryrun.log" 2>&1
tail -5 "$CROSSDISTILL_ROOT/logs/config_dryrun.log"
tail -5 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
