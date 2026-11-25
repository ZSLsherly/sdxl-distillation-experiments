"""Check recorded execution evidence, not merely requested configuration."""
import argparse,json,math
from pathlib import Path
from sdxl_gan_policy_v2 import generator_stage,gan_active,dmd_active,discriminator_stage
ap=argparse.ArgumentParser();ap.add_argument('path');ap.add_argument('--preflight',action='store_true');a=ap.parse_args();p=Path(a.path)
summary=json.loads((p/'summary.json').read_text());status='preflight_complete' if a.preflight else 'training_complete'
assert summary['status']==status
all_g=[]
for rank in range(4):
    s=json.loads((p/f'summary_rank{rank}.json').read_text())
    assert s['status']==status and s['preflight_extra_R1_at_first_D_update']==a.preflight
    rows=[json.loads(x) for x in (p/f'rank{rank}.jsonl').read_text().splitlines()]
    gs=[r for r in rows if r['phase']=='student'];fs=[r for r in rows if r['phase']=='fake_score']
    assert len(gs)==s['student_updates'] and len(fs)==5*len(gs)
    assert all(math.isfinite(r['loss']) and math.isfinite(r['grad_norm']) for r in rows)
    assert s['student_parameter_delta']>0
    for i,g in enumerate(gs):
        v=g['active_variant'];assert g['generator_stage']==generator_stage(i)
        assert g['gan_active']==gan_active(v,i) and g['dmd_active']==dmd_active(v,i)
        if g['gan_active']:assert g['gan_latent_gradient_rms']>0
    ordinal=0
    for f in fs:
        stage=discriminator_stage(f['active_variant'],ordinal)
        assert f['d_fake_stage']==stage
        if stage is None:assert f['d_decode_seconds']==f['d_update_seconds']==0 and 'discriminator_loss' not in f
        else:
            assert f['d_decode_seconds']>0 and f['d_update_seconds']>0
            ordinal+=1
        assert f['fake_score_seconds']>0
        assert f['seconds']+.05>=f['rollout_seconds']+f['d_decode_seconds']+f['d_update_seconds']+f['fake_score_seconds']
    assert ordinal==s['pixel_discriminator_updates']
    counts={str(k):sum(f['d_fake_stage']==k for f in fs) for k in (3,4)}
    assert counts==s['D_stage_counts']
    if a.preflight:
        assert [g['active_variant'] for g in gs]==['dmd','joint34_full','gan4']
        assert counts=={'3':3,'4':7} and ordinal==10
        assert any(f.get('r1_active',False) and f.get('r1',0)>0 for f in fs)
    else:
        v=s['variant'];assert len(gs)==24
        assert s['config']['GAN_modules_loaded']==(v!='dmd')
        assert len({g['cache_index'] for g in gs})==24
        expected={'dmd':{'3':0,'4':0},'gan4':{'3':0,'4':8},'joint4':{'3':0,'4':8},'joint34_matched':{'3':4,'4':4}}[v]
        assert s['GAN_G_stage_counts']==expected
        if v=='joint34_matched':assert counts=={'3':60,'4':60}
        if v=='dmd':assert ordinal==0 and not (p/'discriminator.pt').exists()
        else:assert ordinal==120 and (p/'discriminator.pt').is_file()
        assert (p/f'student_rank{rank}.pt').is_file()
    all_g+=gs
evidence={'status':'passed','ranks':4,'kind':status,'student_updates_per_rank':summary['student_updates'],'D_stage_counts_per_rank':summary['D_stage_counts'],'GAN_G_stage_counts_per_rank':summary['GAN_G_stage_counts'],'unique_student_train_indices_global':len({g['cache_index'] for g in all_g})}
(p/'execution_audit.json').write_text(json.dumps(evidence,indent=2));print(json.dumps(evidence))
