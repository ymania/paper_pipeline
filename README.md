# paper_pipeline

**基于多视图融合的稀疏语义点云生成与BIM轻量化表达** — 端到端论文复现管道

照片 → COLMAP 稀疏重建 → SegFormer 语义分割 → 多视图加权投票 → DBSCAN 实例聚类 → IFC4 BIM 导出

## 快速开始

```bash
# 1. 准备数据：把 COLMAP sparse 输出和图片放到配置路径
# 2. 安装依赖
pip install numpy torch transformers pillow scikit-learn ifcopenshell tqdm

# 3. 一键运行
python run_pipeline.py
```

## 管道步骤

| 步骤 | 脚本 | 论文章节 | 说明 |
|------|------|----------|------|
| 1 | `scripts/1_colmap_parse.py` | §3.2 | COLMAP 稀疏重建解析 |
| 2 | `scripts/2_segformer_infer.py` | §3.3 | SegFormer 语义分割 |
| 3 | `scripts/3_projection.py` | §3.4.1-2 | 3D→2D 投影可见性 |
| 4 | `scripts/4_weighted_voting.py` | §3.4.3-6 | ★ 几何引导多视图加权投票 |
| 5 | `scripts/5_instance_cluster.py` | §3.5 | ★ DBSCAN 实例分割 + OBB |
| 6 | `scripts/6_export_ifc.py` | §3.6 | ★ 语义→IFC4 标准映射导出 |
| 7 | `scripts/7_visualize.py` | 附录 | Plotly 3D 交互可视化 |

## 配置

编辑 `config/paths.json` 设置数据路径，`config/pipeline_params.json` 调整算法参数。

## 输出

- `output/semantic_bim.ifc` — IFC4 BIM 文件，可导入 Revit
- `output/visualization.html` — 3D 交互点云
- `output/stats.json` — 统计摘要

## 环境

- Python 3.9+
- CPU 可运行（i7-12700H 测试通过）
- 全开源工具链（COLMAP / SegFormer / scikit-learn / IfcOpenShell）

## 2026-11-16 阶段成果汇报

[10页PPT、逐页讲稿、真实结果与合成实验说明](reports/2026-11-16/README.md)。该快照明确区分已运行功能与尚未验证结果，原有流水线不因此视为完成独立评价。
