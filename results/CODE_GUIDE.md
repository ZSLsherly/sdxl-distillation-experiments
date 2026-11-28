# 实验与代码对照

每行给出训练或评测的仓库入口，以及该次运行的关键差异。早期多组共用同一训练器，需结合模式列阅读。历史启动脚本可能含多种候选模式；实测设置以本页和随附的配置快照为准。

| 实验 | 对照变量与实际监督 | 配置或实现入口 |
|---|---|---|
| 原始SDXL · 50步 | 公开基础模型，50步、CFG6；提供多步质量参照。 | [`evaluate.py`](../sdxl_coco10k/evaluate.py) |
| 官方DMD2 · 4步 | 公开蒸馏权重，官方四步x₀预测/重新加噪采样、CFG1；训练预算不同。 | [`evaluate.py`](../sdxl_coco10k/evaluate.py) |
| 旧四点 · PCM＋DMD | 999/919/499/199；首步PCM，后续按旧clean目标做DM，作为旧GAN消融基线。 | [`train.py`](../sdxl_parallel/train.py) |
| 旧四点 · 末步纯GAN | 沿用旧四点，末步GAN替代该步DMD；保留前面阶段的监督。 | [`sdxl_gan_train_v3.py`](../sdxl_gan_train_v3.py) |
| 旧四点 · 末步GAN＋DMD | 沿用旧四点，末步同时接受GAN与DMD。 | [`train.py`](../sdxl_parallel/train.py) |
| 旧四点 · 第3/4步GAN＋DMD | 沿用旧四点，GAN覆盖第三、第四步的clean预测。 | [`train.py`](../sdxl_parallel/train.py) |
| 均匀四点 · 旧clean＋末步GAN | 999/749/499/249；PCM×10，学生DM阶段2/3/4轮转，监督clean预测；末步GAN共53次，fake与判别器各800次。 | [`run_full10k_joint4_20261023.py`](../tmp/run_full10k_joint4_20261023.py) |
| 局部边界 · 归一化转向先验 | PCM×1＋随机中段边界DM＋末步DM；加噪仅在509–739/259–489/20–239局部区间。学生用70%均匀＋30%归一化转向曲线，fake均匀并按2→3→4轮转，每G两次；无1%混合、无GAN。 | [`turning_normalized_u_g160.json`](../sdxl_research/configs/turning_normalized_u_g160.json) · [`run_four_gpu_teacher_shard_20261029.py`](../tmp/run_four_gpu_teacher_shard_20261029.py) |
| 局部边界 · 原始转角先验 | 沿用「局部边界 · 归一化转向先验」的局部区间、fake轮转等设置，学生先验改为未按单位距离归一化的原始转角曲线。 | [`turning_raw_angle_g160.json`](../sdxl_research/configs/turning_raw_angle_g160.json) · [`run_four_gpu_teacher_shard_20261029.py`](../tmp/run_four_gpu_teacher_shard_20261029.py) |
| A · 均匀无先验 | 均匀四个推理点；PCM×1＋随机第2/3步带噪边界DM＋末步DM；fake中段/末步配对两次；各合法区间均匀加噪，无GAN、无1%混合。 | [`uniform_control_a800_g160.json`](../sdxl_research/configs/uniform_control_a800_g160.json) · [`run_uniform_after_bc.py`](../sdxl_research/run_uniform_after_bc.py) |
| B · 高噪声先验 | 基于A，在每个合法DM区间把900–979分配30%抽样概率，补集70%；student与fake同分布。 | [`turning_B_quota_a800_g160.json`](../sdxl_research/configs/turning_B_quota_a800_g160.json) · [`run_quota_bc.py`](../sdxl_research/run_quota_bc.py) |
| C · 双端先验 | 中段同B；末步将20–100和900–979的并集合计分配30%，补集70%。 | [`turning_C_quota_a800_g160.json`](../sdxl_research/configs/turning_C_quota_a800_g160.json) · [`run_quota_bc.py`](../sdxl_research/run_quota_bc.py) |
| C＋1%学生→fake | 基于C，从G1起每次学生更新后将1%学生共享参数混入fake；fake阶段embedding及Adam状态不混合。 | [`run_bc_coco_then_winner.py`](../tmp/run_bc_coco_then_winner.py) |
| C＋1% · DM234全选 | 保留PCM；每G为0.5DM2＋0.5DM3＋DM4；fake三个阶段各更新一次。相比「C＋1%学生→fake」同时增加fake预算与中段覆盖。 | [`run_dm234_fake3.py`](../tmp/run_dm234_fake3.py) |
| C＋1% · 首步改DM | 首步PCM替换为边界DM1；每G为DM1＋随机中段DM＋DM4，fake对应三个阶段更新。 | [`prepare_dm1_ablation.py`](../tmp/prepare_dm1_ablation.py) |
| C＋1% · 仅末步学生DM | 保留PCM，只删除学生中段DM；fake仍保留中段一次＋末步一次。 | [`C_mix1_pcm_terminal_dm.json`](../sdxl_research/configs/C_mix1_pcm_terminal_dm.json) |
| A · PCM×10 | 基于A，仅把PCM权重从1改为10；其余采用原A的无先验方案。 | [`run_a_pcm10.py`](../tmp/run_a_pcm10.py) |
| 首步DM＋1% · 无先验 | 基于「C＋1% · 首步改DM」，取消加噪先验；DM1＋随机中段DM＋DM4与三次fake、1%参数混合保留。 | [`run_dm1_uniform_control.py`](../tmp/run_dm1_uniform_control.py) |
| A×10 · DINO/RGB末步GAN | 在「A · PCM×10」基础上，末步clean latent局部解码接DINO＋RGB判别器；学生GAN53次、判别器800次，fake仍320次。 | [`run_a10_patch_gan.py`](../tmp/run_a10_patch_gan.py) |
| A · 中段改预测干净x0 | 在「A · 均匀无先验」基础上，中段DM和fake改用该步预测的clean x0加噪；保持A的中段加噪范围，后续推理仍执行原DDIM跳步。 | [`run_A_mid_x0.py`](../tmp/run_A_mid_x0.py) |
| A×10＋1%学生→fake | 在「A · PCM×10」基础上，从G1起加入1%学生共享权重混入fake；无先验、无GAN。 | [`run_A10_mix1.py`](../tmp/run_A10_mix1.py) |
| A · 中段clean＋全局均匀 | 以A中段clean为基线，仅把中段加噪支持扩为20–979；student/fake同分布。PCM×1，无先验、无混合、无GAN。 | [`run_A_clean_global_n13.py`](../tmp/run_A_clean_global_n13.py) |
| A · 仅末步学生DM | 以A为基线，仅删除学生中段DM，保留PCM×1及末步DM；fake仍中段一次＋末步一次。无先验、无混合、无GAN。 | [`run_A_terminal_only_n13.py`](../tmp/run_A_terminal_only_n13.py) |
| A×10 · 末步低噪声30% | 以A×10为基线，仅末步student/fake改为20–100占30%、101–979占70%；中段不变，无混合、无GAN。 | [`run_A_terminal_prior_n13.py`](../tmp/run_A_terminal_prior_n13.py) |
| A · clean全局＋高噪声30% | 以A中段clean＋全局均匀为基线，第2/3/4步student/fake均以900–979占30%、20–899占70%抽样；PCM×1不变，无混合、无GAN。采用修复随机流后的v2。 | [`run_clean_high30_v2_n14.py`](../tmp/run_clean_high30_v2_n14.py) |
| C＋1% · 额外去噪GAN | 在「C＋1%学生→fake」基础上增加训练专用末步去噪分支：终点detach后在249加噪，学生额外预测，整图VAE＋SAM判别器；正式评测仍四步。 | [`train.py`](../tmp/h20_mb32_code/train.py) |
| 自训 DMD2，原生四步随机采样 | LAION 衍生 10K、batch 64、G160/F800；采用 clean DM 和 DMD2 判别器。训练与公开权重的预算、配方不同。评分使用原生随机重新加噪采样。 | [`train_matched.py`](../output/sdxl-crossdistill/dmd2_matched_20261119/code/train_matched.py) |

## 50K 单遍训练

| 实验 | 学生与 fake 的监督 | 实现与实际配置 |
|---|---|---|
| old_A | 首步 PCM×1＋随机中段边界 DM＋末步 DM。中段 student/fake 从带噪边界再加噪。 | [`queue50.py`](../output/sdxl-crossdistill/a50k_train60_20261124/deploy/queue50.py) · [运行配置](configs/old_A.json) |
| old_A_no_middle_DM | 仅去掉旧 A 的学生中段 DM；仍做中段和末步两次 fake 更新。 | [`after_singlepass_no_middle.py`](../output/sdxl-crossdistill/a50k_train60_20261124/no_middle/after_singlepass_no_middle.py) · [运行配置](configs/old_A_no_middle_DM.json) |
| clean_x0_A | 保留学生中段 DM；student 和 fake 均从该步的 clean x₀ 预测加独立噪声，fake 输入 detach。 | [`train.py`](../results/clean50k/train.py) · [运行配置](configs/clean_x0_A.json) |

四个存档来自同一次训练。20K、30K、40K、50K 分别对应 G313、G470、G627、G784，里程碑的尾批单独处理。G 表示学生优化器更新次数。新 clean 实现保留原始训练模块；默认数据根目录改为相对路径，可用 SDXL_ROOT 指定。

## 小规模与诊断实验

| 实验 | 实现入口 | 读数 |
|---|---|---|
| 学习率 × PCM 权重（4 组） | [`sdxl_hparam_train.py`](../sdxl_hparam_train.py) | G12，固定 16 条 |
| 早期局部 GAN（4 组） | [`sdxl_gan_train_v2.py`](../sdxl_gan_train_v2.py) | 追加 G24，固定 16 条 |
| 原始 4 步、多步与公开蒸馏模型 | [`evaluate.py`](../sdxl_coco10k/evaluate.py) | 早期 16 条与正式评测分列 |
| 教师转向曲线 | [`sdxl_turning_resample.py`](../sdxl_turning_resample.py) | 128 条提示词 × 4 种子 |
| 逐阶段教师替换 | [`sdxl_stage_intervention.py`](../sdxl_stage_intervention.py) | 32 条 × 6 个分支 |
| microbatch / 卸载 | [`run_microbatch16_20261023.py`](../tmp/run_microbatch16_20261023.py) | 显存与秒/G |
| H20 判别器 / VAE 分批 | [`train.py`](../tmp/h20_mb32_code/train.py) | 完整周期、显存及 OOM |
| 末步高20%＋低10%的 D 组 | [`run_interval_D.py`](../tmp/run_interval_D.py) | G9 停止 |
| 官方配方 DMD2 对齐尝试 | [`train_matched.py`](../output/sdxl-crossdistill/dmd2_official_recipe_20261121/code/train_matched.py) | G106 中断，缺完整评分 |
| Wan 教师轨迹复现 | [`analyze36.py`](../output/dymd_prior_8_20261101/analyze36.py) | 8 个输入共 36 条轨迹 |
| clean-x₀ 去中段 20K | [归档启动配置](configs/clean_no_mid_20k.json) · [共用 clean 实现](clean50k/train.py) | 从 clean A 派生，保留 PCM 和中段 fake，仅去掉学生中段 DM。 未取得完整评分。 |
| clean-x₀ PCM×10 50K | [归档启动配置](configs/clean_pcm10_50k.json) · [共用 clean 实现](clean50k/train.py) | 从 clean A 派生，首步 PCM 的系数由 1 调至 10。 未取得完整评分。 |
