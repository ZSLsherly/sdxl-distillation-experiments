from pathlib import Path
import json,hashlib,zipfile,shutil
r=Path.cwd();o=r/'output/sdxl-crossdistill/project_audit_20261112'
fix=o/'fixes';fix.mkdir(exist_ok=True);shutil.copy2(r/'tmp/run_terminal_low30.py',fix/'run_terminal_low30.py')
findings=[{'priority':'P1','kind':'confirmed code error','title':'Completion verifier requires events absent from inherited trainer','before':'evidence/run_terminal_low30_before.py:54','trainer':'sdxl_research/train.py:955','fixed':'tmp/run_terminal_low30.py:54','status':'locally fixed and reproduced; not deployed'}, {'priority':'P1','kind':'operational state risk','title':'Local and deployed low30 recipes not reconciled','status':'remote verification unavailable; do not count as trained'}, {'priority':'P2','kind':'scientific design','title':'Terminal fake low-noise regression is weighted by min(1,10*sigma^2)','source':'sdxl_research/boundary.py:60','status':'verified algebra; unchanged'}, {'priority':'P2','kind':'scientific design','title':'DM coverage and fake update count changed together','source':'tmp/patch_dm234_fake3.py','status':'report effect as combined recipe'}, {'priority':'P2','kind':'diagnostic limitation','title':'Original C lacks fixed-t heldout fake history for a matched tracking claim','source':'sdxl_research/fake_tracking.py:49','status':'cannot reconstruct from student-only checkpoint'}]
(o/'findings.json').write_text(json.dumps(findings,ensure_ascii=False,indent=2),encoding='utf-8')
results=json.loads((o/'experiment_results.json').read_text(encoding='utf-8'))
index={'audit_date':'2026-11-12','completed_historical_self_trained':17,'public_references':2,'pending_latest_verification':['A_pcm10_mix1','normalized_u_g160','raw_angle_g160'],'prepared_not_confirmed_run':['C_mix1_pcm_dm4_fake4_low30_g160','pcm_dm4_weight_ablation_20261109/dm4_x1','pcm_dm4_weight_ablation_20261109/dm4_x2'],'results':'experiment_results.json','findings':'findings.json','sources':'source_inventory.json','test_counts':{'core':21,'weighted_dm':3},'live_limitation':'3090 SSH timeout twice on 2026-11-12; report retains verified historical results'}
(o/'project_index.json').write_text(json.dumps(index,ensure_ascii=False,indent=2),encoding='utf-8')
# Check the report assets and all numeric rows before packaging.
assert all((o/x).is_file() for x in ['report.html','quality_tradeoffs.png','fake_low_noise_weight.png','项目审计报告.txt'])
assert len(results)==19
for x in results:assert (r/'output/sdxl-crossdistill'/x['source']).is_file(),x['source']
with zipfile.ZipFile(o.parent/'SDXL项目审计_20261112.zip','w',zipfile.ZIP_DEFLATED) as z:
 for p in o.rglob('*'):
  if p.is_file() and 'test_tmp' not in p.parts:z.write(p,'project_audit_20261112/'+str(p.relative_to(o)))
print('PACKAGE_VERIFIED',len(results),'rows')
