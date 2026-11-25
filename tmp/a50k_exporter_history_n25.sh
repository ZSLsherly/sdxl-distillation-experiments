systemctl show gpu-exporter -p FragmentPath -p WorkingDirectory -p StandardOutput -p StandardError
python3 - <<'PY'
import subprocess
for service in ['gpu-exporter','zabbix-agent']:
 p=subprocess.run(['journalctl','-u',service,'--since','2026-11-25 03:40','--no-pager','-n','150','-o','cat'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 lines=p.stdout.decode(errors='replace').splitlines()
 print(service,'exit',p.returncode,'recent_lines',len(lines),'no_entries',any('No entries' in x for x in lines),'pid_keyword',sum('pid' in x.lower() for x in lines),'permission_hint',bool(p.stderr))
PY
for n in 0 1 2 3;do p=$(cat /data5/shenglin.zeng/a50k_20261124/outputs/A50K_no_middle_singlepass/pid_rank$n);ps -p "$p" -o pid=,etime=,comm=;done
