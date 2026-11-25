import json
from pathlib import Path
p=Path('sdxl_pilot_protocol.json');r=json.loads(p.read_text(encoding='utf-8'))
r['sources']['DyMD']='https://arxiv.org/html/2609.31349v1'
r['teacher_turning_followup']={'user_feedback':'Investigate a substantially earlier crossover; t899 already has much of the composition fixed.', 'primary_metric':'DyMD Eq6 angle between unit teacher velocities divided by physical-noise interval; C=identity image adaptation', 'data':'16 held-out LAION prompts,2seeds=32trajectories', 'controls':'Fine stride5 on16trajectories;FP32compute originalFP16weights on8trajectories', 'candidates_for_images':[979,949,919,899], 'training_control':'All four queued optimizer/PCM-weight trials retain crossover749; not a chosen optimum'}
p.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')
