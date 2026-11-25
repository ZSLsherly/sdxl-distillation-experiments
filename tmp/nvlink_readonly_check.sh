nvidia-smi nvlink --help
nvidia-smi nvlink --status
nvidia-smi nvlink --capabilities
nvidia-smi topo -p2p n
for p in /sys/module/nvidia/parameters/NVreg_NvLinkDisable /proc/driver/nvidia/params; do
  if test -r "$p"; then grep -iE 'nvlink|peer|fabric' "$p"; fi
done
ls /proc/driver/nvidia-nvlink 2>/dev/null
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv
