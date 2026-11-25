set -eu
. /root/sdxl_research/env.sh
export CUDA_VISIBLE_DEVICES=''
export PYTHONPATH="$SDXL_ROOT/code/sdxl_research"
python - <<'PY'
import json
from pathlib import Path
import boundary as b
root=Path('/root/sdxl_research');run=root/'outputs/turning_BC_paired_quota_20261102/B'
c=json.loads((run/'config.json').read_text());assert c['pair_fake_stages'] and c['match_fake_prior'] and c['variant']=='dmd_only'
rows=[json.loads(s) for s in (run/'rank0.jsonl').read_text().splitlines()];inputs=[r for r in rows if r['phase']=='fake_boundary_input']
knots=[999,749,499,249,-1];ranges=b.phase_ranges(knots,10,979)
for r in inputs:
 k=b.critic_stage(r['fake_update'],True,c['seed']);assert r['stage']==k and r['boundary_t']==knots[k]
 assert r['renoise_range']==list(ranges[k])
 if k<4:assert r['boundary_vs_final_rms']>0
 else:assert r['boundary_vs_final_rms']==0
fs=[r for r in rows if r['phase']=='fake'];gs=[r for r in rows if r['phase']=='student_done']
for r in gs:
 i=r['g']-1;assert [x['fake_stage'] for x in fs[2*i:2*i+2]]==[r['stage'],4]
result={'config_paired':True,'matching_time_distribution':True,'GAN_enabled':c['discriminator_enabled'],'inputs':inputs,'completed_G':len(gs),'completed_fake':len(fs),'latest_G':{k:gs[-1][k] for k in ['g','stage','cycle_seconds','grad_norm']} if gs else None}
(run.parent/'paired_boundary_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
PY
