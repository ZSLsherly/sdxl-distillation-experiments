source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python - <<'PY'
from smoke_config import make_config
c=make_config()
print('before',c.defaults,c.model.config.fsdp_shard_size,c.optimizer)
from imaginaire.utils.config_helper import override
c=override(c,None)
print('after',c.defaults,c.model.config.fsdp_shard_size,c.optimizer)
PY
tail -15 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
