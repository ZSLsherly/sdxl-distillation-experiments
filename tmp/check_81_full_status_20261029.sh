set -eu
root=/data3/shenglin.zeng/sdxl_turning_ablation_20261028
date -Is
if test -d "$root";then
 ls -lh "$root/transfers" "$root/tmp"
 find "$root/outputs" -maxdepth 4 -type f \( -name status.json -o -name benchmark.json -o -name workflow_complete.json -o -name workflow.log \) -print
 for f in "$root"/outputs/*/workflow.log;do if test -f "$f";then echo "$f";tail -n 8 "$f";fi;done
 for p in runtime/bin/python env/bin/python datasets/parallel_10k_seed20261018/prepared.pt models/sdxl-base-1.0/unet/config.json;do test ! -e "$root/$p" || ls -lh "$root/$p";done
 df -h "$root" | tail -1
fi
pgrep -u "$(id -u)" -af '[t]urning_batch64_workflow.py|[s]moke_four_gpu_20261028|[s]etup_81_experiment' || true
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
