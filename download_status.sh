ROOT=/data5/shenglin.zeng
printf 'TORCH\n'; tail -5 "$ROOT/logs/install_torch.log"
printf '\nTOOLS\n'; tail -3 "$ROOT/logs/download-tools.log"
printf '\nMODELS\n'; tail -8 "$ROOT/logs/download_models.log"
printf '\nGIT\n'; du -sh "$ROOT/src/SparkDiffusion/.git"; ls "$ROOT/src/SparkDiffusion"
