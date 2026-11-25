set -u
hostname
nvidia-smi --query-gpu=index,name,uuid,pci.bus_id,pci.device_id,pci.sub_device_id,vbios_version,memory.total,driver_version --format=csv
nvidia-smi -q
nvidia-smi topo -m
nvidia-smi nvlink --status
command -v lspci >/dev/null && lspci -Dnn | grep -i nvidia
python - <<'PY'
from pathlib import Path
for p in sorted(Path('/sys/bus/pci/devices').iterdir()):
    try:
        if (p/'vendor').read_text().strip()!='0x10de':continue
        fields={k:(p/k).read_text().strip() for k in ['vendor','device','subsystem_vendor','subsystem_device','class','revision'] if (p/k).exists()}
        if not fields.get('class','').startswith(('0x03','0x12')):continue
        for k in ['current_link_speed','current_link_width','max_link_speed','max_link_width']:
            if (p/k).exists():fields[k]=(p/k).read_text().strip()
        print('PCI_SYSFS',p.name,fields)
    except (OSError,PermissionError):pass
PY
