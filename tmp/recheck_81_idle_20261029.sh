set -eu
for i in 1 2;do
 date -Is
 nvidia-smi --id=1,2,4,5 --query-gpu=index,memory.free,utilization.gpu --format=csv,noheader,nounits
 nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid,used_gpu_memory --format=csv,noheader,nounits
 if test "$i" = 1;then sleep 10;fi
done
