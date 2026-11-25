ctl="$HOME/.discovery/bin/discovery-ctl"
cfg=/tmp/laion-discovery-904.b9IQDB/config
"$ctl" --config-dir "$cfg" dataset info --dataset Ashly/LAION-10K --format json
"$ctl" --config-dir "$cfg" dataset list --dataset Ashly/LAION-10K --version main --format json
"$ctl" --config-dir "$cfg" dataset upload-file --help
python3 -c 'from pathlib import Path; p=Path("/tmp/laion-discovery-904.b9IQDB/config"); print("auth-config-present", p.exists() and any(p.iterdir()))'
