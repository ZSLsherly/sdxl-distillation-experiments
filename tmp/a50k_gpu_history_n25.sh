date -Is
nvidia-smi --query-gpu=index,uuid,accounting.mode,accounting.buffer_size,utilization.gpu --format=csv,noheader
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
nvidia-smi --help-query-accounted-apps | head -65
nvidia-smi --query-accounted-apps=gpu_uuid,pid,time,max_memory_usage --format=csv,noheader 2>&1 | head -40
systemctl list-units --type=service --all --no-pager | grep -Ei 'dcgm|prometheus|netdata|atop|telegraf|ganglia|gpu|collectd|grafana|zabbix' || true
ls -ld /var/log/atop /var/log/pcp/pmlogger /var/lib/prometheus /var/log/dcgm /var/log/sysstat 2>/dev/null || true
ss -ltHn | awk '$4 ~ /:(9090|9400|19999|3000|9100|8086)$/ {print $4}'
cd /data5/shenglin.zeng/a50k_20261124
find logs -maxdepth 1 -type f -printf '%f\n'
for f in logs/A50K_G320.log logs/A50K_singlepass.log logs/A50K_no_middle_singlepass.log; do echo "$f";grep -im1 'Process [0-9].*memory in use' "$f" | cut -c1-580 || true;done
python3 - <<'PY'
import json
from pathlib import Path
for p in Path('outputs').glob('*/rank0.jsonl'):
 rows=[json.loads(l) for l in p.read_text().splitlines() if l.endswith('}')]
 rows=[r for r in rows if 'timestamp' in r]
 if rows:print(p.parent.name,rows[0]['timestamp'],rows[-1]['timestamp'])
PY
