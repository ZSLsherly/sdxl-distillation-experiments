# SDXL 少步蒸馏实验

本仓库保存 SDXL 图像蒸馏的训练与评测源码，包含不同生成阶段的分布匹配、首步 PCM、末步 Pixel GAN，以及教师转向先验和 fake 网络更新策略的消融。历史实验使用了不同的预测目标。复核某一组时，请同时查看该组的训练入口、配置和调度器。

| 目录 | 用途 |
| --- | --- |
| [sdxl_research](sdxl_research/) | 共用训练代码、阶段选择、损失、数据批处理和实验配置 |
| [50K 实验](output/sdxl-crossdistill/a50k_train60_20261124/) | 50K 训练实际部署的代码、去中段对照以及里程碑评测 |
| [sdxl_parallel](sdxl_parallel/) | 早期并行训练实现和 GAN 消融 |
| [sdxl_coco10k](sdxl_coco10k/) | 固定 COCO10K 的采样、评测和报告脚本 |
| [_deployment_archives](_deployment_archives/) 和 [_result_archives](_result_archives/) | 从历史部署包、实验包还原的代码和配置 |

环境版本可查 [environment.json](sdxl_research/environment.json)。运行前按所选实验配置模型、数据和输出目录。历史脚本保留原来的绝对路径和启动参数，复现时请对应调整。训练权重、数据集和登录凭据由使用者另行提供。

## 参考实现

- [DMD2](https://github.com/tianweiy/DMD2) 及其 [上游许可](attribution/DMD2-LICENSE.md)
- [CDM](https://github.com/byliutao/CDM)
- [SparkDiffusion](https://github.com/AlibabaResearch/SparkDiffusion)
- [PhysisForcing](https://github.com/DAGroup-PKU/PhysisForcing)
- [BasicSR](https://github.com/XPixelGroup/BasicSR) 及其 [上游许可](benchmark_vendor/LICENSE.txt)
