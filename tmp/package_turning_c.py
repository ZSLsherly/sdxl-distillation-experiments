import json
import tarfile
from pathlib import Path

root=Path(__file__).resolve().parent.parent
source=root/'sdxl_turning_c'
with tarfile.open(root/'tmp/turning_c_20261020.tar.gz','w:gz') as tar:
    for p in source.iterdir():
        if p.is_file():
            tar.add(p,arcname=p.name)
out=root/'output/sdxl-crossdistill/turning_c_20261020'
out.mkdir(parents=True,exist_ok=True)
protocol={
    "experiment":"C: teacher-turning mixture for student DMD re-noising",
    "date":"2026-10-20",
    "baseline":"outputs/gan_stage_joint4_4gpu/joint4_20261019T122237_4gpu_mb16",
    "initialization":"untouched SDXL Base; no warm-start from G160",
    "teacher_cfg":8,"student_cfg":1,"student_schedule":[999,919,499,199,-1],
    "student_stage_cycle":[2,3,4],"gan_stages":[4],"pcm_weight":10,
    "gan_weight":0.005274229682981968,"student_lr":1e-6,"fake_lr":1e-6,"d_lr":1e-4,
    "batch":64,"micro_batch_per_gpu":16,"world_size":4,"rollout_batch":16,
    "fake_updates_per_student":5,"max_student_updates":160,
    "save_every_updates":40,"eval_every_updates":2,
    "final_checkpoint":"full student/fake/D/optimizer/counters/RNG",
    "data_dir":"datasets/parallel_10k_seed20261018",
    "only_algorithm_change":"student t_D: 0.7 Uniform[20,919] + 0.3 teacher turning prior restricted and normalized on same support",
    "fake_sampling":"unchanged Uniform[0,999]; fake reweighting not enabled",
    "importance_correction":False,
    "rng":"separate deterministic stream; all original noise/PCM/crop/flip inputs preserved",
    "turning_distribution":json.loads((source/'turning_distribution.json').read_text()),
    "limitations":["Turning prior is fixed across images; not full video DyMD",
        "Existing 4-image monitoring is diagnostic only, not efficacy evidence",
        "Shared GPU compute affects wall-clock comparison; other users' processes left untouched"]
}
(out/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf-8')
html="""<!doctype html><html lang="zh"><meta charset="utf-8"><title>C 实验实施配置</title>
<style>body{font:17px/1.8 system-ui;max-width:900px;margin:45px auto;padding:20px;color:#172339}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccd5e1;padding:9px;text-align:left}h1{font-size:28px}</style>
<h1>C：老师转向引导的 DMD 加噪采样</h1><p>2026-10-20。准备启动的独立实验；本文不是已完成效果报告。</p>
<p>沿用已有末步 DMD+GAN 基线，从原始 SDXL 初始化。唯一算法变化为学生 DMD 再加噪时间的概率分布。</p>
<table><tr><th>设置</th><th>值</th></tr>
<tr><td>推理轨迹</td><td>999 → 919 → 499 → 199 → 干净终点</td></tr>
<tr><td>监督</td><td>首段 TM；第2、3步 DMD；第4步 DMD+GAN</td></tr>
<tr><td>学生 DMD 时间</td><td>70% 原均匀分布 + 30% 老师转向平方先验；范围20–919</td></tr>
<tr><td>先验</td><td>复用32条 SDXL 老师轨迹的平滑全图转向；在支持域内归一化；不做重要性反向修正</td></tr>
<tr><td>概率变化</td><td>平均 timestep：469.5 → 556.26；t≥800 概率：13.33% → 27.40%</td></tr>
<tr><td>fake</td><td>仍用0–999均匀噪声，5次 fake/D 更新对应1次学生更新</td></tr>
<tr><td>数据与算力</td><td>原 parallel_10k_seed20261018 数据，4×A100，每卡16，总batch64</td></tr>
<tr><td>训练预算</td><td>160次学生更新；每40次存学生权重，最终存完整续训状态</td></tr>
<tr><td>学习率</td><td>student/fake 1e-6；D 1e-4</td></tr>
<tr><td>损失权重</td><td>PCM 10；GAN 0.005274229682981968</td></tr>
<tr><td>随机性</td><td>独立时间采样流；保持原始latent噪声、PCM噪声、裁剪和翻转一致</td></tr></table>
<p>先执行分布、跨批次确定性、原随机输入不变性与原有CPU测试，再启动正式训练。旧实验源码保持不变。</p>
<p>每2次更新的小样本图片只用于监控。C是否有效仍需统一质量、语义和多样性评测；本次不启动A/B重训。</p>
<p><a href="protocol.json">完整机器可读配置</a></p></html>"""
(out/'protocol.html').write_text(html,encoding='utf-8')
print(out/'protocol.html')
