"""Evaluate both trained students using the C_mix1 manifest and A800 evaluator."""
import fcntl
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import time

from run_bc_coco_then_winner import on_idle_gpu, env

root = Path(os.environ['SDXL_ROOT'])
outputs = root/'outputs'
base = outputs/'bc_winner_late_mix_20261102'
reference = base/'coco10k_C_mix1'
inputs = Path(os.environ['SDXL_WORK'])/'bc_texture_inputs'
code = root/'tmp/bc_coco_winner_20261102/code'
out = outputs/'c_mix1_dm_pair_coco10k'
arms = {'DM234_fake3':outputs/'c_mix1_dm234_fake3_20261103',
        'DM1_mid4_fake3':outputs/'c_mix1_first_dm_fake3_g160'}

def read(path):
    return json.loads(path.read_text())

def save(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    tmp.replace(path)

def timing(path):
    with (path/'rank0.jsonl').open() as stream:
        gs = [r for line in stream if (r:=json.loads(line))['phase']=='student_done']
    assert [r['g'] for r in gs] == list(range(1,161))
    def when(row):
        return datetime.strptime(row['timestamp'],'%Y%m%dT%H%M%S%z')
    wall = (when(gs[-1])-when(gs[0])).total_seconds()+gs[0]['cycle_seconds']
    cycles = sum(r['cycle_seconds'] for r in gs)
    assert 0 < cycles <= wall+2
    return dict(train_wall_hours=wall/3600, train_cycle_hours=cycles/3600)

def score(folder, arm):
    result=read(folder/'complete.json')
    s=result['scores'][arm]
    assert result['n_per_model']==s['n']==10000 and all(math.isfinite(s[k]) for k in ('FID','Patch_FID'))
    return {k:s[k] for k in ('FID','Patch_FID')}

def evaluate(arm, training, manifest):
    evidence=read(training/'complete.json')
    assert (evidence['G'],evidence['fake'])==(160,480)
    ck=Path(evidence['checkpoint'])
    result=evaluate_checkpoint(arm, ck, read(training/'model/config.json'), manifest, out)
    return dict(**result,**timing(training/'model'))

def evaluate_checkpoint(arm, ck, config, manifest, output, run_worker=on_idle_gpu):
    assert read(ck/'cpu_validation.json')['strict_load'] and (ck/'shapes_rank0.json').is_file()
    updates=read(ck/'complete.json')['student_updates']
    assert config['schedule']==manifest['models']['C_mix1']['config']['schedule']
    with (ck/'student_rank0.pt').open('rb') as f:
        digest=hashlib.file_digest(f,'sha256').hexdigest()
    spec=dict(checkpoint=str(ck),world_size=1,student_updates=updates,config=config,
              shards=[dict(name='student_rank0.pt',sha256=digest)])
    dest=output/arm
    (dest/'features').mkdir(parents=True,exist_ok=True)
    real=dest/'features/real'
    if not real.is_symlink(): real.symlink_to((reference/'features/real').resolve(),target_is_directory=True)
    assert real.resolve()==(reference/'features/real').resolve() and real.is_dir()
    record=dict(manifest,models={arm:spec})
    mp=dest/'manifest.json'
    if mp.exists(): assert read(mp)==record
    else: save(mp,record)
    if not (dest/'complete.json').exists():
        opts=['--root',str(inputs),'--out',str(dest)]
        entry=[sys.executable,'-u',str(code/'evaluate.py')]
        run_worker(entry+['worker',*opts,'--methods',arm,'--batch','4'],dest/'worker.log')
        with (dest/'metrics.log').open('a') as log:
            subprocess.run(entry+['metrics',*opts],cwd=code,env={**env,'CUDA_VISIBLE_DEVICES':''},
                           stdout=log,stderr=subprocess.STDOUT,check=True)
    return score(dest,arm)

def main():
    out.mkdir(exist_ok=True)
    with (out/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        m=read(reference/'manifest.json')
        assert m['n']==len(m['samples'])==10000 and len(m['visual_indices'])==128
        assert all(r['index']==i and r['seed']==20261020+i for i,r in enumerate(m['samples']))
        assert (inputs/'models/sdxl-base-1.0/model_index.json').is_file()
        assert (inputs/'models/metrics/clean_fid/inception-2015-12-05.pt').is_file()
        assert (reference/'features/real').is_dir()
        baseline=dict(**score(reference,'C_mix1'),**timing(base/'late_mix'))
        results={'C_mix1':baseline}
        save(out/'baseline.json',baseline)
        print('BASELINE',baseline,'WAIT_FOR_BOTH_TRAININGS',flush=True)
        while not all((p/'complete.json').is_file() for p in arms.values()):
            time.sleep(30)
        for arm,training in arms.items():
            result=evaluate(arm,training,m)
            result['delta_vs_C_mix1']={k:v-baseline[k] for k,v in result.items()}
            result['extra_training_percent']=100*(result['train_wall_hours']/baseline['train_wall_hours']-1)
            results[arm]=result
            save(out/'comparison.json',results)
            lines=['固定 COCO2014 10K，FID 与中心 299×299 Patch FID 越低越好。','训练墙钟从第一次更新开始到 G160 更新完成，含中途看图；不含排队、加载、最终存档及 COCO 跑分。',
                   '组名 | FID | Patch FID | 训练小时 | 比基准多耗时 | ΔFID | ΔPatch FID']
            for key,v in results.items():
                d=v.get('delta_vs_C_mix1',{})
                lines.append(f"{key} | {v['FID']:.4f} | {v['Patch_FID']:.4f} | {v['train_wall_hours']:.3f} | {v.get('extra_training_percent',0):+.1f}% | {d.get('FID',0):+.4f} | {d.get('Patch_FID',0):+.4f}")
            lines.append('两组都比基准多一次 fake 更新；首步 DM 组也移除了 PCM。差值反映整组配置在 G160 的收益。')
            (out/'比较结果.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
        save(out/'complete.json',results)
        print('COMPLETE',flush=True)

if __name__=='__main__':
    main()

