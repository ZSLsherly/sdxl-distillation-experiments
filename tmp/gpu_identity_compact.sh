nvidia-smi --query-gpu=index,name,pci.bus_id,pci.device_id,pci.sub_device_id,vbios_version,memory.total,power.default_limit,power.max_limit --format=csv
for d in /sys/bus/pci/devices/0000:01:00.0 /sys/bus/pci/devices/0000:61:00.0 /sys/bus/pci/devices/0000:a1:00.0 /sys/bus/pci/devices/0000:c1:00.0; do
  echo "$d"
  for key in vendor device subsystem_vendor subsystem_device revision current_link_speed current_link_width; do
    printf '%s=' "$key"
    cat "$d/$key"
  done
done
nvidia-smi topo -m
nvidia-smi nvlink --status
df -h /tmp /data5
