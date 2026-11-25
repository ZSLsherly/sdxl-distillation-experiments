ROOT=/data5/shenglin.zeng
nohup bash "$ROOT/install_deps.sh" > "$ROOT/logs/install_deps.log" 2>&1 < /dev/null &
printf 'DEPS_PID=%s\n' "$!"
printf '\nDOWNLOAD PROGRESS\n'
du -sh "$ROOT/models" "$ROOT/cache/pip"
tail -4 "$ROOT/logs/download_models.log"
tail -4 "$ROOT/logs/install_torch.log"
