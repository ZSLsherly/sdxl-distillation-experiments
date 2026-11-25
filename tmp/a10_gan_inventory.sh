set -eu
. /root/sdxl_research/env.sh
ls "$SDXL_ROOT/models"
ls -l "$SDXL_ROOT/models/dinov2-small" || true
df -h /root "$SDXL_WORK"
nvidia-smi --query-compute-apps=pid --format=csv,noheader
cd "$SDXL_ROOT/outputs/A_pcm10_g160/model/source"
grep -nE 'sam_path|train_d =|if boundary_mode:|use_gan =|for fake_repeat|gan_value|G_PCM|D_fake_stage|cached_final|cfg_prediction|optimiz' train.py | tail -55

