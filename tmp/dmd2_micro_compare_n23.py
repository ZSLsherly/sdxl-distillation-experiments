import json,statistics
from pathlib import Path
r=Path('output/sdxl-crossdistill/dmd2_official_recipe_4gpu_20261123')
out={}
for micro,folder in [(8,'短测证据'),(16,'micro16短测证据')]:
    ranks=[[json.loads(x) for x in (r/folder/f'rank{k}.jsonl').read_text(encoding='utf-8').splitlines()] for k in range(4)]
    complete=json.loads((r/folder/'complete.json').read_text(encoding='utf-8'))
    assert complete['strict_load'] and complete['finite'] and complete['G']==4 and complete['F']==20
    out[micro]={'G':4,'F':20,'micro':micro,'accumulation':64//(4*micro),
        'peak_GiB':max(x['peak_GiB'] for rows in ranks for x in rows),
        'training_seconds':statistics.mean(sum(x['seconds'] for x in rows) for rows in ranks),
        'complete_elapsed_seconds':complete['elapsed_seconds'],'strict_load':True,'finite':True}
ratio=out[8]['training_seconds']/out[16]['training_seconds']
out['throughput_ratio_16_over_8']=ratio
(r/'microbatch对比.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
(r/'microbatch实测与选择.txt').write_text(
    '2026-11-23 同一G4/F20预算、global batch64，四rank平均训练更新耗时（不含评估/导出）。\n'
    '原始日志已rclone取回并核验SHA256。两组均通过实际保存和严格加载。\n'
    'micro16原先仅被39GiB保守资源门槛挡住，并非OOM或数值失败；实测后改为42GiB资源门槛，选择16正式训练。\n'
    '最终导出前释放Adam状态以降低保存峰值；依用户确认仅保存最终推理权重。\n'
    +json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
