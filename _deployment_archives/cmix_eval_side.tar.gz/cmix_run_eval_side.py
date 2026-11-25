import fcntl,json,math,os,subprocess,sys
from pathlib import Path
from run_bc_coco_then_winner import code,env,inputs,on_idle_gpu,job,evalout,save,sha,verify
out=job/'coco10k_C_mix1'
out.mkdir(exist_ok=True)
with (out/'eval.lock').open('a') as lock:
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    c=json.loads((job/'late_mix/config.json').read_text())
    assert c['variant']=='dmd_only' and c['fake_student_mix']==.01 and c['fake_mix_start_g']==1
    assert 'quota_C_' in c['turning_profile'] and c['schedule']==[999,749,499,249,-1]
    dest=Path(verify(job/'late_mix'))
    v=json.loads((dest/'cpu_validation.json').read_text());assert v['strict_load'] and v['finite']
    manifest=json.loads((evalout/'manifest.json').read_text())
    assert manifest['n']==len(manifest['samples'])==10000 and len(set(manifest['visual_indices']))==128
    method='C_mix1'
    manifest['models']={method:dict(checkpoint=str(dest),world_size=1,student_updates=160,config=c,shards=[dict(name='student_rank0.pt',sha256=sha(dest/'student_rank0.pt'))])}
    path=out/'manifest.json'
    if path.exists():assert json.loads(path.read_text())==manifest
    else:save(path,manifest)
    real=out/'features/real';real.parent.mkdir(exist_ok=True)
    ref=(evalout/'features/real').resolve(strict=True)
    if not real.exists():real.symlink_to(ref,target_is_directory=True)
    assert real.resolve(strict=True)==ref
    base=[sys.executable,'-u',str(code/'evaluate.py')]
    opts=['--root',str(inputs),'--out',str(out)]
    cpu=dict(env,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
    print('VERIFIED C+1% checkpoint; waiting for idle GPU',flush=True)
    if not (out/'complete.json').exists():
        done=out/'worker_0_done.json'
        if not done.exists():on_idle_gpu(base+['worker',*opts,'--methods',method,'--batch','4'],out/'worker.log')
        with (out/'metrics.log').open('a') as f:
            subprocess.run(base+['metrics',*opts],cwd=code,env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
    scores=json.loads((out/'complete.json').read_text())['scores'][method]
    assert scores['n']==10000 and all(math.isfinite(scores[k]) for k in ('FID','Patch_FID'))
    print('FID_COMPLETE',scores,flush=True)
    with (out/'semantic.log').open('a') as f:
        subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('cmix_semantic.py'))],env=cpu,stdout=f,stderr=subprocess.STDOUT,check=True)
    sem=json.loads((out/'semantic.json').read_text())
    assert sem['n']==128 and len(sem['per_case'][method]['alignment'])==128
    baseline=json.loads((Path(os.environ['SDXL_WORK'])/'semantic_abc_20261103/results.json').read_text())
    assert sem['cases']==baseline['cases'] and sem['revision']==baseline['revision']
    assert max(abs(x-y) for x,y in zip(sem['per_case']['C']['alignment'],baseline['per_case']['C']['alignment']))<1e-3
    print('SEMANTIC_COMPLETE',sem['metrics'],sem['differences'],flush=True)
    save(out/'evaluation_complete.json',dict(scores=scores,semantics=sem['metrics'],paired_difference=sem['differences']))
