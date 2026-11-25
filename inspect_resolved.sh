source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python - <<'PY'
from smoke_config import make_config
from imaginaire.utils.config_helper import override
c=override(make_config(),None)
print('NET',c.model.config.net)
print('FSDP',c.model.config.fsdp_shard_size,'RFMAX',c.model.config.rf_t_max)
print('OPTIMIZER',c.optimizer)
print('CALLBACKS',c.trainer.callbacks.keys())
assert not c.model.config.net.use_rola_attn
assert c.model.config.fsdp_shard_size==4
assert c.model.config.loss_scale_dmd==1 and c.model.config.loss_scale_tdm==0
assert c.optimizer.optim_type=='adamw'
print('CONFIG_VERIFIED')
PY
tail -8 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
