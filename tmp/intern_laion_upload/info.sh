ctl="$HOME/.discovery/bin/discovery-ctl"
"$ctl" login --info 2>&1 | head -c 900
"$ctl" dataset search --help
"$ctl" dataset info --help
