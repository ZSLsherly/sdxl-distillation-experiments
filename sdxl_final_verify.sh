source /data5/shenglin.zeng/env.sh
root=/data5/shenglin.zeng/sdxl_crossdistill
python - <<'PY'
import json
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill')
for name in ['teacher_stages','teacher_turning','pilot_assessment']:
 p=root/f'logs/{name}.exit';assert p.read_text().strip()=='0';print(name,'EXIT_0')
assert json.loads((root/'outputs/hparam_pilot/audit.json').read_text())['status']=='passed'
report=root/'outputs/hparam_pilot/report.html';assert report.is_file()
(root/'outputs/hparam_pilot/completion.json').write_text(json.dumps({'status':'complete','training_trials':4,'student_updates_per_trial':12,'fake_updates_per_trial':60,'teacher_turning_main_trajectories':32,'assessment_images_per_method':16,'audit':'passed','report':str(report),'convergence':'not established; pilot outputs remain blurry'},indent=2))
print('PILOT_COMPLETE_REPORT_SAVED')
PY
nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
