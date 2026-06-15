"""
基于几何先验的多视图加权投票语义融合 — 论文 §3.4.4-3.4.6

核心思想:
  同一 3D 点在不同视角下可能获得不一致的语义标签。
  通过距离权重 (w_dist) 和角度权重 (w_angle) 综合投票，
  选出最可靠的标签。

权重函数:
  w_ij = α · w_dist + β · w_angle
  w_dist = 1 / (1 + d_ij)            — 越近越可靠
  w_angle = |n_i · v_ij|              — 正面观测最可靠

投票函数:
  V_i(l) = Σ_j w_ij · 1[S_j(u,v) = l]
  l_i* = argmax_l V_i(l)

时间复杂度: O(n · k)（n=点数，k=平均可见视图数）
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def compute_distance_weight(
    point: np.ndarray,
    camera_center: np.ndarray
) -> float:
    """
    距离权重 — 越近越可靠 (§3.4.4)

    w_dist = 1 / (1 + d)
    """
    d = np.linalg.norm(point - camera_center)
    return 1.0 / (1.0 + d)


def compute_angle_weight(
    normal: np.ndarray,
    viewing_dir: np.ndarray
) -> float:
    """
    角度权重 — 正面观测最可靠 (§3.4.4)

    w_angle = |cos(θ)| = |n · v|

    直观解释: 相机正对物体表面时分割结果最可靠。
    """
    cos_angle = np.abs(np.dot(normal, viewing_dir))
    return np.clip(float(cos_angle), 0.0, 1.0)


def weighted_vote(
    points: np.ndarray,
    normals: np.ndarray,
    visibility: dict,
    cameras: dict,
    images: dict,
    semantic_maps: List[np.ndarray],  # 每张图像的语义分割结果 (H×W, int)
    image_id_to_index: Dict[int, int],  # COLMAP image_id → semantic_maps 索引
    num_classes: int = 150,
    alpha: float = 0.5,
    beta: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    对每个 3D 点进行多视图加权投票，分配语义标签。

    参数:
        points:          (N, 3) 点云坐标
        normals:         (N, 3) 法向量
        visibility:      {point_idx: {'cam_indices':[...], 'px_coords':[...], 'depths':[...]}}
        cameras:         {camera_id: {K, R, t, width, height}}
        images:          {image_id: {R, t, C}}
        semantic_maps:   [np.ndarray(H,W)], 每张图像的 SegFormer 输出
        image_id_to_index:  COLMAP 的 image_id → semantic_maps 的数组索引
        num_classes:     ADE20K 类别数 (150)
        alpha:           距离权重系数
        beta:            角度权重系数 (α+β=1)

    返回:
        labels:          (N,) 语义标签 (ADE20K class id)
        confidences:     (N,) 标签置信度 [0,1]
    """
    N = len(points)
    # 投票矩阵太大 (N×150)，使用逐点稀疏累积
    labels = np.full(N, -1, dtype=np.int32)
    confidences = np.zeros(N, dtype=np.float32)
    vote_counts = np.zeros(N, dtype=np.int32)

    processed = 0
    skipped_no_vis = 0
    skipped_no_sem = 0

    for idx, vis_info in visibility.items():
        if idx >= N:
            continue

        cam_indices = vis_info['cam_indices']
        px_coords = vis_info['px_coords']
        depths = vis_info['depths']

        if not cam_indices:
            skipped_no_vis += 1
            continue

        pt = points[idx]
        normal = normals[idx]

        # 对每个可见视图累积投票
        votes = {}  # {class_id: accumulated_weight}

        for i, img_id in enumerate(cam_indices):
            # 获取该图像的语义分割结果
            sem_idx = image_id_to_index.get(img_id)
            if sem_idx is None or sem_idx >= len(semantic_maps):
                continue

            u, v = px_coords[i]
            ui, vi = int(round(u)), int(round(v))

            sem_map = semantic_maps[sem_idx]
            H, W = sem_map.shape
            if ui < 0 or ui >= W or vi < 0 or vi >= H:
                continue

            class_id = int(sem_map[vi, ui])

            # 计算权重
            img = images[img_id]
            cam_center = img['C']  # 相机中心世界坐标

            w_dist = compute_distance_weight(pt, cam_center)
            
            viewing_dir = cam_center - pt
            viewing_dir_norm = np.linalg.norm(viewing_dir)
            if viewing_dir_norm > 1e-10:
                viewing_dir = viewing_dir / viewing_dir_norm
            w_angle = compute_angle_weight(normal, viewing_dir)

            weight = alpha * w_dist + beta * w_angle
            votes[class_id] = votes.get(class_id, 0.0) + weight

        if not votes:
            skipped_no_sem += 1
            continue

        # 选择最高票标签
        best_class = max(votes, key=votes.get)
        total_weight = sum(votes.values())
        confidence = votes[best_class] / total_weight if total_weight > 0 else 0.0

        labels[idx] = best_class
        confidences[idx] = confidence
        vote_counts[idx] = len(cam_indices)  # 可见视图数（非唯一种类数）

        processed += 1
        if processed % 5000 == 0:
            logger.info(f"加权投票: {processed} / ~{len(visibility)}")

    logger.info(f"投票完成: {processed} 个点获得标签, "
                f"跳过 (无可见视图={skipped_no_vis}, 无语义={skipped_no_sem})")
    
    valid = labels >= 0
    if valid.sum() > 0:
        logger.info(f"平均置信度: {confidences[valid].mean():.4f}")
        logger.info(f"唯一标签数: {len(np.unique(labels[valid]))}")

    # 多视图统计 — 按可见视图数量分层分析
    multi_view_mask = vote_counts > 1
    single_view_mask = vote_counts == 1
    n_single = single_view_mask.sum()
    n_multi = multi_view_mask.sum()
    
    # 视图数分布
    if n_multi > 0:
        multi_confs = confidences[multi_view_mask]
        n_disagree = (multi_confs < 0.99).sum()
        logger.info(f"多视图分析: 单视图={n_single}点, 多视图={n_multi}点")
        logger.info(f"  视图数分布: 2视图={(vote_counts==2).sum()}, 3视图={(vote_counts==3).sum()}, 4+视图={(vote_counts>=4).sum()}")
        logger.info(f"  多视图中有标签分歧: {n_disagree}点 ({n_disagree/n_multi*100:.1f}%)")
        if n_disagree > 0:
            logger.info(f"  分歧点平均置信度: {multi_confs[multi_confs < 0.99].mean():.4f}")
            logger.info(f"  分歧点置信度中位数: {np.median(multi_confs[multi_confs < 0.99]):.4f}")
        logger.info(f"  多视图点平均置信度: {multi_confs.mean():.4f}")
    else:
        logger.info(f"多视图分析: 所有标记点均仅从1个视角可见")
        logger.info(f"  注: COLMAP稀疏重建中track长度偏短是正常现象")

    return labels, confidences
