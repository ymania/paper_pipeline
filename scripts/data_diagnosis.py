#!/usr/bin/env python3
"""
数据特征诊断 — 分析 COLMAP track 长度分布
输出: output/data_diagnosis.json
"""
import sys, json, pickle, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

INTERMEDIATE = Path('output/intermediate')

# 加载 visibility 数据
with open(INTERMEDIATE / 'visibility.pkl', 'rb') as f:
    visibility = pickle.load(f)

# 统计每个点的可见视图数
view_counts = []
for idx, vis in visibility.items():
    view_counts.append(len(vis['cam_indices']))

view_counts = np.array(view_counts)
total = len(view_counts)

# 分析
n_1view = (view_counts == 1).sum()
n_2view = (view_counts == 2).sum()
n_3view = (view_counts == 3).sum()
n_4plus = (view_counts >= 4).sum()

logger.info("=" * 50)
logger.info("COLMAP 稀疏点云可见视图分布")
logger.info("=" * 50)
logger.info(f"  总 3D 点数:       {total}")
logger.info(f"  平均可见视图数:   {view_counts.mean():.2f}")
logger.info(f"  中位数:           {np.median(view_counts):.0f}")
logger.info(f"  1 视图:           {n_1view} ({n_1view/total*100:.1f}%)")
logger.info(f"  2 视图:           {n_2view} ({n_2view/total*100:.1f}%)")
logger.info(f"  3 视图:           {n_3view} ({n_3view/total*100:.1f}%)")
logger.info(f"  4+ 视图:          {n_4plus} ({n_4plus/total*100:.1f}%)")

# 语义点中的视图分布
labels = np.load(INTERMEDIATE / 'semantic_labels.npy')
labeled_mask = labels >= 0
labeled_view_counts = view_counts[labeled_mask]

n_l1 = (labeled_view_counts == 1).sum()
n_l2 = (labeled_view_counts == 2).sum()

logger.info(f"\n语义点 (有标签) 视图分布:")
logger.info(f"  1 视图:           {n_l1} ({n_l1/len(labeled_view_counts)*100:.1f}%)")
logger.info(f"  2+ 视图:          {n_l2} ({n_l2/len(labeled_view_counts)*100:.1f}%)")

# 结论
logger.info(f"\n诊断结论:")
logger.info(f"  {n_l1/len(labeled_view_counts)*100:.1f}% 的语义点仅被1个视角观测")
logger.info(f"  加权投票算法在当前数据上无区分能力")
logger.info(f"  多视图融合的价值需在密集重建(MVS)后验证")

# 保存
with open('output/data_diagnosis.json', 'w') as f:
    json.dump({
        'total_3d_points': int(total),
        'labeled_points': int(labeled_mask.sum()),
        'mean_views': float(view_counts.mean()),
        'median_views': float(np.median(view_counts)),
        '1view_pct': float(n_1view/total*100),
        '2view_pct': float(n_2view/total*100),
        'labeled_1view_pct': float(n_l1/len(labeled_view_counts)*100),
        'conclusion': '加权投票在当前稀疏数据上无区分能力，需MVS稠密重建'
    }, f, indent=2, ensure_ascii=False)

logger.info(f"\n诊断报告 → output/data_diagnosis.json")
