set -eu
ps -eo pid,args | grep -E '[p]ip( |$)|[c]onda (install|update)|[t]rain.py|[t]orch.distributed.run' || true
ls -ld /root/.cache/pip /tmp/pip-* 2>/dev/null || true
du -h --max-depth=2 /root/miniconda3/lib/python3.8/site-packages 2>/dev/null | sort -h | tail -n 18
ls -lh /tmp/pip-unpack-* 2>/dev/null | head -n 45
