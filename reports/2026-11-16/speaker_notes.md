## 1. 建筑三维重建与窗户候选提取

本次汇报聚焦已有影像上的窗户候选检测与关联。项目背景为基于无人机影像的3D数字孪生，但本组person-hall影像的采集来源尚未独立核验，不将其称为已验证的无人机采集实验。来源：reports/2026-11-16/assets/context.jpg；REPORT.md。

## 2. 研究范围与工程目标

目标是提供附带来源视图和未知状态的窗户候选，减少多张照片之间人工查找的工作。尚无人工时间测量，不声称节省百分比。完整mesh、实测尺寸、IFC构件正确性均不在本轮已验证范围。

## 3. 已有基础与工程修复

来源：output/continuation_20261028/report.md；output/annotation_closure_20261028/fusion_audit.json。修复包括畸变投影、图像ID/相机ID对应、点ID与数组行一致性等。此页指已有局部诊断资产，不是本轮从影像重建的完整建筑。

## 4. 原有分割未区分窗户与建筑整体

来源：reports/2026-11-16/assets/anchor_semantic.jpg；evidence/baseline_results.json。B5缓存与已检查原图显示窗户被归为building。人工点标注存在墙选项缺陷，因此不使用其宏IoU作为最终研究成绩。

## 5. 自动检测获得可用的窗户候选

来源：reports/2026-11-16/assets/1123_candidates.jpg；evidence/detections.json。Grounding DINO tiny，固定revision，提示词a window.；阈值0.30，文本阈值0.25，NMS 0.50。https://huggingface.co/IDEA-Research/grounding-dino-tiny 。检测框与原分割掩膜任务不同，不能直接声称同指标提升。

## 6. SfM 对应点支持跨视图关联

来源：evidence/associations.json。规则：至少3个共有track；交集/较小点集至少0.20；每组每个视图最多一个候选；至少3个视图称为多视图候选。属于开发原型，不是新颖性证明或独立实例准确率。

## 7. 多视图一致仍会保留误检

来源：assets/1124_candidates.jpg；evidence/associations.json。顶部百叶状开口被window提示检测，4个视图重复出现。是否计为通风百叶应由固定类别定义决定。保留此困难负例，未事后提高阈值隐藏它。

## 8. 合成实验：可见性筛选有适用条件

来源：evidence/synthetic/summary.json、assumptions.json。模拟单一平面几何、6个窗区、5视图；每条件20个噪声种子。5%与35%可见性误差由人为翻转生成，非真实估计器精度。图为均值，SD保存在JSON；种子不是独立建筑。

## 9. 当前证据的边界

来源：REPORT.md；DATA.md；现有人工标注100条，墙选项缺陷使类间指标需重新解释。没有删除原始标注或将building自动改为wall。模型和方法均使用同一已查看开发立面，未开展盲测。

## 10. 下一阶段的验证重点

汇报结论：从工程可运行推进到独立有效性验证。可使用已有公开标注减少人工逐点审核；需要核对许可、类别定义及跨建筑划分。Scan2LoD3 已有可见性+语义融合+建筑开口重建，不能仅将模块组合当创新。https://openaccess.thecvf.com/content/CVPR2023W/PCV/papers/Wysocki_Scan2LoD3_Reconstructing_Semantic_3D_Building_Models_at_LoD3_Using_Ray_CVPRW_2023_paper.pdf