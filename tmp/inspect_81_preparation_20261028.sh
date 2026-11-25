set -eu
id -un
pwd
df -h -x tmpfs -x devtmpfs
for d in /data1 /data2 /data3 /data4 /data5 /data6 /home; do
 if test -d "$d"; then ls -ld "$d" "$d/shenglin.zeng" 2>/dev/null || true; fi
done
command -v python || true
for base in "$HOME" /data1/shenglin.zeng /data2/shenglin.zeng /data5/shenglin.zeng; do
 if test -d "$base"; then find "$base" -maxdepth 4 \( -name prepared.pt -o -name 'sdxl-base-1.0' -o -name 'conda-meta' -o -name pyvenv.cfg \) -print 2>/dev/null | head -50; fi
done
nvidia-smi --id=1,2,4,5 --query-gpu=index,uuid,memory.free,utilization.gpu --format=csv,noheader,nounits
nvidia-smi --id=1,2,4,5 --query-compute-apps=gpu_uuid,pid --format=csv,noheader,nounits
