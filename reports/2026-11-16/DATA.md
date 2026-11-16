# 数据说明

- assets/：开发场景中的选定展示图，原始采集许可与UAV来源在本次未独立核验；本仓库不对源图作新增许可声明，不能将其称作已发布的UAV基准数据集。
- evidence/detections.json：固定版本模型、原图尺寸及SHA256、原始框和得分、保留候选。
- evidence/associations.json：候选组、关联边、track ID及支持点范围；范围不是窗的实测几何。
- evidence/baseline_results.json、fusion_audit.json：旧分割局部诊断及三基线重放记录。
- evidence/synthetic/：假设参数与模拟指标，全部不得作为真实建筑实验。
- 未上传：原始个人标注记录、完整图像场景、模型权重、浏览器缓存和机器环境信息。JSON中的机器路径已替换为符号路径，图像哈希仍可用于校对。
- publication_manifest.json：本成果包文件哈希。

当前数据足以展示运行结果与失败案例，不足以证明方法泛化、实例准确率、真实尺寸或完整mesh质量。
