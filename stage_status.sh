ROOT=/data5/shenglin.zeng
date -Is
tail -8 "$ROOT/logs/segmented_download.log"
printf '\nFILES\n'
ls -lh "$ROOT/models/Wan2.1-T2V-1.3B/"*.pth "$ROOT/models/Wan2.1-T2V-1.3B/"*.safetensors "$ROOT/models/Wan2.1-T2V-1.3B/"*.segmented 2>/dev/null
printf '\nCONFIG\n'
grep -E 'fsdp_shard_size|rf_t_max|optim_type|loss_scale_dmd|loss_scale_tdm|max_iter' "$ROOT/outputs/sparkdiffusion/CrossDistill_Smoke/wan1.3B_dense_pcm_dmd_smoke/config.yaml"
