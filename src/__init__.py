"""
paper_pipeline — 论文 "基于多视图融合的稀疏语义点云生成与BIM轻量化表达" 复现代码

模块与论文章节对应:
  colmap_io         → §3.2 三维重建模块 (COLMAP 数据解析)
  projection        → §3.4.1-3.4.2 投影模型与可见性
  normal_estimation → §3.4.3 几何先验 (PCA法向量)
  weighted_voting   → §3.4.4-3.4.6 几何引导加权投票
  clustering        → §3.5 DBSCAN 实例分割 + OBB
  ifc_export        → §3.6 IFC 映射与导出
  coord_transform   → COLMAP→IFC 坐标系转换

用法:
  from src.colmap_io import read_cameras_txt
  from src.projection import compute_visibility
  from src.weighted_voting import weighted_vote
  from src.clustering import cluster_semantic_points
  from src.ifc_export import export_ifc
"""
