set -eu
umask 077
work=$(mktemp -d "${TMPDIR:-/tmp}/laion-discovery-904.XXXXXX")
echo "AUTH_WORK=$work"
nohup "$HOME/.discovery/bin/discovery-ctl" --config-dir "$work/config" login >"$work/login.log" 2>&1 </dev/null &
echo "AUTH_PID=$!"
sleep 2
head -c 3500 "$work/login.log"
