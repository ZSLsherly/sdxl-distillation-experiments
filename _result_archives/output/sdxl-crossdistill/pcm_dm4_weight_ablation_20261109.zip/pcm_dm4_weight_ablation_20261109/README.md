# SDXL · PCM + 末步学生 DM 的权重消融

本实验在 C＋1% 的「PCM＋仅末步学生DM，中段fake保留」上比较末步DM系数。原组的G160成绩为 COCO10K FID 22.1288、中心299 Patch FID 28.0241。原配置另有等时延长，本包两组统一在 G160 结束，只比较 G160。

| 配置 | 学生目标 | 每轮 fake 更新 | 末步加噪先验 |
|---|---|---|---|
| configs/dm4_x1.json | PCM + DM4 | 一个中段，随后末步 | 原 C 双端并集 30% |
| configs/dm4_x2.json | PCM + 2 × DM4 | 同上 | 同上 |

两份 JSON 仅 terminal_dm_weight 不同。种子20261017、10K训练数据、999/749/499/249、PCM×1、无GAN、fake_weight_max=10、G1起1%学生混入fake均按历史配方。学生不对中段反传。G160/F320，单卡 micro16 × 累积4，有效 batch64。所有参数梯度先求和，然后按原阈值1裁剪，再做一次学生Adam更新。系数2恢复DM的名义总权重，不能保证每个样本的实际梯度范数等于原C＋1%。

## 在已有 A800 项目环境运行

解压到独立目录，进入含 launch.py 的目录。使用原 SDXL 训练环境，原模型位于 $SDXL_ROOT/models/sdxl-base-1.0，10K 缓存位于 $SDXL_ROOT/datasets/parallel_10k_seed20261018（prepared.pt、manifest.json）。本包附有环境版本清单和原C先验文件。留足最终学生权重（约4.8 GiB）、日志和固定评估图所需空间。

    export SDXL_ROOT=/root/sdxl_research
    python test_ablation.py
    python launch.py --root "$SDXL_ROOT" --config configs/dm4_x2.json --gpus 0 --run c_mix1_pcm_dm4_x2_g160 --execute

去掉 --execute 可只打印命令。需要重跑带相同诊断的×1时，使用 dm4_x1.json 和独立名称 c_mix1_pcm_dm4_x1_g160。配置中的原始先验路径由包内 launch.py 解析到包内同版本 JSON。不要用相同名称覆盖既有结果。

输出到 $SDXL_ROOT/outputs/<run>/。训练保存最终BF16学生推理权重并进行固定图评估。完整COCO10K使用项目已有评测程序另行评测。

## 如何读梯度日志

rank0.jsonl 的 student_done 中 terminal_dm_loss 是未加权的末步代理损失，dmd_loss 是用于学生更新的加权DM；本组二者的比值应为2（原始项为零时例外）。PCM单列；fake损失记录沿用原协议。

pcm_dm_gradient_probe 在即将执行G1、40、80、120、160的学生更新前测量。使用相同4个训练样本、相同噪声，记录原始样本ID。该记录使用裁剪前、全体学生参数的梯度；不代表整批64的实际梯度。weighted_dm_over_pcm 是加权末步DM对PCM的范数比，cosine 是两者夹角余弦。另存原始DM范数和加权DM范数。若任一范数为零，无法定义的比值或余弦写为null。探测使用独立的随机状态，不更新参数，并在正式累积前清空探测梯度。两组均包含同一探测开销。

本包在本地验证了参数解析、CPU梯度缩放、固定探测范数与余弦、160轮fake阶段配对及压缩包内容。未启动GPU训练。
