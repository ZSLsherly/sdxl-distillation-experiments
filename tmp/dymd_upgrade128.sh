set -eu
B=/data1/shenglin.zeng/dymd_prior_8_20261101
python3 - <<'PY'
import json,pathlib
b=pathlib.Path('/data1/shenglin.zeng/dymd_prior_8_20261101')
assert json.loads((b/'status.json').read_text())['stage']=='waiting_for_model_weights'
assert not list((b/'profiles').glob('trajectory_*.jsonl'))
PY
pid=$(cat "$B/controller.pid")
case "$(ps -p "$pid" -o args=)" in *dymd_prior_8_20261101/controller.py*) kill -TERM "$pid";; *) exit 2;; esac
mkdir -p "$B/original8_protocol"
cp "$B/protocol.json" "$B/controller.py" "$B/profile_official.py" "$B/prepare_run.py" "$B/original8_protocol/"
echo '{"stage":"upgrading_to_128","target_trajectories":128}' > "$B/status.json"
