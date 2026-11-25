ROOT=/data5/shenglin.zeng
printf 'INSTALL LOG\n'
tail -8 "$ROOT/logs/install_torch.log"
printf '\nREPO FILES\n'
ls "$ROOT/src/SparkDiffusion/scripts/distill" 2>/dev/null
find "$ROOT/src/SparkDiffusion/sparkdiffusion/configs/experiments" -type f 2>/dev/null
printf '\nSKILL AND 1.3B LAUNCHER\n'
cat "$ROOT/src/SparkDiffusion/scripts/distill/wan2.1_1.3b_t2v_480p.sh" 2>/dev/null
