set -eu
source /data5/shenglin.zeng/env.sh
cd "$CROSSDISTILL_ROOT"
python -m scripts.train --config=smoke_config.py --dryrun > logs/config_dryrun.log 2>&1
python - <<'PY'
from pathlib import Path
from omegaconf import OmegaConf
p=Path('/data5/shenglin.zeng/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/config.yaml')
c=OmegaConf.load(p)
assert c.model.config.fsdp_shard_size==4
assert c.model.config.loss_scale_dmd==1 and c.model.config.loss_scale_tdm==0
assert c.optimizer.optim_type=='adamw'
assert not c.model.config.net.get('use_rola_attn',False)
assert c.model.config.rf_t_max==.94
assert 'smoke_metrics' in c.trainer.callbacks
print('RESOLVED_CONFIG_VERIFIED')
PY
cd "$CROSSDISTILL_ROOT/src/SparkDiffusion"
git diff --stat
printf '\nDOWNLOAD\n'
tail -6 "$CROSSDISTILL_ROOT/logs/segmented_download.log"
