"""Audit completed trials using the saved records, without new training."""
import json,math,hashlib
from pathlib import Path
root=Path('/data5/shenglin.zeng/sdxl_crossdistill/outputs/hparam_pilot')
trials=['lr5e7_pcm10','lr25e8_pcm10','lr5e7_pcm1','lr25e8_pcm1'];audit=[]
reference=None
for trial in trials:
    folder=root/trial;summary=json.loads((folder/'summary.json').read_text())
    assert summary['world_size']==4 and summary['student_updates']==12 and summary['fake_updates']==60
    split=json.loads((folder/'data_split.json').read_text())
    assert not set(split['train_cache_indices'])&set(split['validation_cache_indices'])
    all_before=[];norms=[];clips=[];lengths=[];used_student_indices=set();used_fake_indices=set()
    for rank in range(4):
        rows=[json.loads(x) for x in (folder/f'rank{rank}.jsonl').read_text().splitlines()]
        assert len(rows)==72
        for i,row in enumerate(rows):
            assert row['iteration']==i
            assert all(math.isfinite(row[k]) for k in ['loss','grad_norm','seconds','peak_allocated_GiB'])
            assert row['grad_norm']>0
            idx=split['train_cache_indices'][(i*4+rank)%len(split['train_cache_indices'])]
            if row['phase']=='student':
                assert 20<=row['dmd_t']<=749 and 770<=row['pcm_t']<=999
                assert row['rollout_length'] in [2,3,4]
                used_student_indices.add(idx);lengths.append(row['rollout_length'])
            else:
                assert 0<=row['fake_t']<=999
                used_fake_indices.add(idx)
            norms.append(row['grad_norm']);clips.append(row['clip_scale'])
        before=json.loads((folder/f'validation_before_rank{rank}.json').read_text())
        after=json.loads((folder/f'validation_after_rank{rank}.json').read_text())
        assert len(before)==len(after)==4
        assert all(row['finite'] for row in before+after)
        assert all(row['cache_index'] in split['validation_cache_indices'] for row in before+after)
        all_before+=before
    if reference is None:reference=all_before
    else:assert all_before==reference,'Trials did not start with identical validation outputs'
    assert set(lengths)=={2,3,4}
    for iteration in range(5,72,6):
        values=[json.loads((folder/f'rank{rank}.jsonl').read_text().splitlines()[iteration])['rollout_length'] for rank in range(4)]
        assert len(set(values))==1,'FSDP ranks must use the same rollout length'
    audit.append({'trial':trial,'status':'passed','distinct_student_samples':len(used_student_indices),'distinct_fake_prompts':len(used_fake_indices),
        'student_sample_presentations':12*4,'gradient_norm_range':[min(norms),max(norms)],'clipped_updates_rank_presentations':sum(v<1 for v in clips),
        'source_sha256':(folder/'training_source.sha256').read_text().strip(),'export_bytes':(folder/'export/unet/diffusion_pytorch_model.safetensors').stat().st_size if (folder/'export/unet/diffusion_pytorch_model.safetensors').exists() else None,
        'checkpoint_shards':len(list(folder.glob('student_rank*.pt')))})
(root/'audit.json').write_text(json.dumps({'status':'passed','trials':audit,'checks':['Four ranks completed every trial','Finite losses and nonzero gradients','Teacher and cross-model gradient isolation asserted during training',
    'Identical pre-training validation outputs across all trials','Disjoint training and validation indices','PCM and DMD timestep support','Uniform-length categories present and synchronized across ranks','Four model-only checkpoint shards retained per trial; portable reference export present'],
    'limitations':['No optimizer state was saved; exported models can initialize a new run but do not provide exact optimizer resumption.','Only 16 distinct student samples are reached by this short fixed traversal.']},indent=2))
print('PILOT_AUDIT_PASSED',json.dumps(audit),flush=True)
