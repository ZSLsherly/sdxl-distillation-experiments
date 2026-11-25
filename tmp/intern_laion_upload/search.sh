ctl="$HOME/.discovery/bin/discovery-ctl"
cfg=/tmp/laion-discovery-904.b9IQDB/config
"$ctl" --config-dir "$cfg" dataset search --keywords LAION --format json | head -c 5000
