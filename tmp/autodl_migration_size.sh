set -eu
root=/root/autodl-tmp/sdxl_research
du -sh "$root"/*
du -sb "$root"/env "$root"/runtime "$root"/models "$root"/datasets "$root"/code
cat "$root/env/pyvenv.cfg"
ls -l "$root/env/bin/python" "$root/env/bin/python3" "$root/env/bin/python3.12"
command -v rsync || true
test ! -e /root/sdxl_research || ls -ld /root/sdxl_research
ps -eo pid,args | grep -E '[t]rain.py|[b]ash setup.sh|[t]orch.distributed.run' || true
