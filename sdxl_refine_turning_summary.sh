source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
cp "$root/outputs/hparam_pilot/teacher_turning/result.json" "$root/outputs/hparam_pilot/teacher_turning/result_initial_interpolation.json"
python "$root/src/sdxl_summarize_turning.py" --statistics-only
