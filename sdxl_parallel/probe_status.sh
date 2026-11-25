cd /data5/shenglin.zeng/sdxl_crossdistill
date -Is
for name in parallel_mb16_adamcpu parallel_mb32_adamcpu; do
  echo "$name"
  if [ -f "logs/${name}.log" ]; then grep -E 'LOAD|UPDATE|EVALUATED|Error:|OutOfMemory|BENCHMARK' "logs/${name}.log" | tail -n 8; fi
  if [ -f "logs/${name}.exit" ]; then cat "logs/${name}.exit"; fi
done
nvidia-smi --query-gpu=index,memory.used,utilization.gpu,power.draw --format=csv,noheader
