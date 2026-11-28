# 测量与可视化

| 图 | 归档来源 |
|---|---|
| [coco_three_models.png](figures/coco_three_models.png) | A50K_clean_x0_20261125/coco_milestones/coco_three_models.png |
| [fid_patchfid_scatter.png](figures/fid_patchfid_scatter.png) | loss_19_20_20261126/fid_patchfid_scatter.png |
| [conditional_prior.png](figures/conditional_prior.png) | loss_19_20_20261126/conditional_prior.png |
| [loss_19_20_20261126_student.png](figures/loss_19_20_20261126_student.png) | loss_19_20_20261126/student.png |
| [loss_19_20_20261126_fake.png](figures/loss_19_20_20261126_fake.png) | loss_19_20_20261126/fake.png |
| [loss_50k_20261126_student.png](figures/loss_50k_20261126_student.png) | loss_50k_20261126/student.png |
| [loss_50k_20261126_fake.png](figures/loss_50k_20261126_fake.png) | loss_50k_20261126/fake.png |
| [full.jpg](figures/full.jpg) | five_paired_samples_20261126/full.jpg |
| [center.jpg](figures/center.jpg) | five_paired_samples_20261126/center.jpg |
| [turning512.png](figures/turning512.png) | teacher_turning_resample_20261026/results512/turning512.png |
| [stage_sensitivity.png](figures/stage_sensitivity.png) | stage_intervention_20261026/stage_sensitivity.png |
| [training_curves.png](figures/training_curves.png) | hparam_pilot/training_curves.png |
| [all_scores.png](figures/all_scores.png) | all_experiments_20261116/all_scores.png |
| [fake_bc_separate.png](figures/fake_bc_separate.png) | a800_BC_paired_quota_20261102/loss_diagnostics/fake_bc_separate.png |
| [student_by_stage.png](figures/student_by_stage.png) | a800_BC_paired_quota_20261102/loss_diagnostics/student_by_stage.png |
| [holdout_overview.jpg](figures/holdout_overview.jpg) | a800_BC_paired_quota_20261102/final_comparison/holdout_overview.jpg |
| [benchmark_curves.png](figures/benchmark_curves.png) | benchmark_final_20261020/benchmark_curves.png |
| [texture_native_crops.png](figures/texture_native_crops.png) | gan_stage_pilot_v2/texture_native_crops.png |

## 数值证据

[指标与诊断快照](measurements.jsonl.gz) 和 [训练曲线](training_curves.jsonl.gz) 采用 gzip 压缩的 JSON Lines。解压后每行包含归档来源和数值。部分运行有多个不同时间的快照，或同一日志的备份；按 source 选择后再统计。指标记录保留数值和实验分支，训练记录保留阶段、更新、损失、梯度和峰值显存；来源名称中的计算节点已匿名化。历史快照可能来自短测、中断或失效版本。正式结果以主表所列独立归档为准。

- [stage_intervention_20261026/metrics.csv](diagnostics/stage_intervention_20261026_metrics.csv)
- [teacher_turning_resample_20261026/results512/curves.csv](diagnostics/teacher_turning_resample_20261026_curves.csv)
- [benchmark_final_20261020/summary.csv](diagnostics/benchmark_final_20261020_summary.csv)
本次归档含 203 份可提取数值的测量快照和 32 份训练日志，共 31,077 条日志记录。[逐文件索引](evidence_index.json)
