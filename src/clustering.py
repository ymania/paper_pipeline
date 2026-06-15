"""
基于 DBSCAN 的语义实例分割 — 论文 §3.5

流程:
  1. 对每个语义类别分别做 DBSCAN 聚类
  2. 噪声过滤
  3. PCA-OBB 包围盒拟合 (§3.5.4)

OBB 参数:
  - 中心: μ = (1/|P|) Σ p
  - 协方差: Σ, 特征值分解得主轴 U
  - 半长度: s_k = max(p-μ)^T u_k - min(p-μ)^T u_k
"""
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from sklearn.cluster import DBSCAN
from sklearn.neighbors import LocalOutlierFactor
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OBB:
    """定向包围盒"""
    center: np.ndarray        # (3,) 中心
    axes: np.ndarray          # (3, 3) 主轴，行向量
    half_extents: np.ndarray  # (3,) 半长度
    class_id: int = -1
    instance_id: int = -1
    class_name: str = ""
    num_points: int = 0

    def to_dict(self) -> dict:
        return {
            'center': self.center.tolist(),
            'axes': self.axes.tolist(),
            'half_extents': self.half_extents.tolist(),
            'class_id': self.class_id,
            'instance_id': self.instance_id,
            'class_name': self.class_name,
            'num_points': self.num_points
        }

    @property
    def dimensions(self) -> np.ndarray:
        """长宽高"""
        return self.half_extents * 2

    @property
    def volume(self) -> float:
        return np.prod(self.dimensions)


def compute_obb(points: np.ndarray) -> OBB:
    """
    PCA-OBB 拟合 (§3.5.4)

    参数:
        points: (M, 3) 同一实例的点集

    返回:
        OBB 对象
    """
    if len(points) < 3:
        raise ValueError(f"至少需要 3 个点，实际: {len(points)}")

    center = points.mean(axis=0)
    centered = points - center

    # 协方差矩阵 + 特征值分解
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 按特征值降序 (主轴 → 次轴 → 辅轴)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order].T  # (3,3) 行向量

    # 确保右手坐标系
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1

    # 将点变换到主轴坐标系，计算半长度
    transformed = centered @ axes.T
    min_vals = transformed.min(axis=0)
    max_vals = transformed.max(axis=0)
    half_extents = (max_vals - min_vals) / 2.0

    return OBB(
        center=center,
        axes=axes,
        half_extents=half_extents
    )


def filter_outliers_lof(
    points: np.ndarray,
    n_neighbors: int = 20,
    contamination: float = 0.1
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    LOF 离群点预过滤 — 在 DBSCAN 之前剔除显著离群点 (§3.5.1 改进)

    参考: 张曾一 (2025) §3.5, 使用局部离群因子降低噪声对聚类干扰

    参数:
        points:       (N, 3) 点云
        n_neighbors:  LOF 邻域大小 (默认 20)
        contamination: 预期离群比例 (默认 0.1 = 10%)

    返回:
        filtered_points:  (M, 3) 过滤后的点云
        inlier_mask:      (N,) bool 内点掩码
        n_removed:        被剔除的离群点数
    """
    if len(points) < n_neighbors:
        logger.warning(f"点数 ({len(points)}) < n_neighbors ({n_neighbors})，跳过 LOF")
        return points, np.ones(len(points), dtype=bool), 0

    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=False
    )
    outlier_labels = lof.fit_predict(points)
    inlier_mask = outlier_labels == 1
    n_removed = (~inlier_mask).sum()

    logger.info(
        f"LOF 离群过滤: {n_removed}/{len(points)} 点剔除 "
        f"({n_removed/len(points)*100:.1f}%), "
        f"剩余 {inlier_mask.sum()} 点"
    )

    return points[inlier_mask], inlier_mask, n_removed


def align_to_principal_axes(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    PCA 主方向对齐 — 将点云旋转到主轴对齐坐标系 (§3.5.1 改进)

    参考: 张曾一 (2025) §3.3, 桥梁主方向校准;
    葛健 (2024) §4.3.2, OBB 主轴提取

    使点云的主轴对齐到世界坐标轴 (X, Y, Z)，
    便于后续 DBSCAN 的 eps 参数调优和 OBB 尺寸计算。

    参数:
        points: (N, 3) 原始点云

    返回:
        aligned_points: (N, 3) 对齐后的点云
        rotation:       (3, 3) 旋转矩阵 R (aligned = points @ R)
    """
    if len(points) < 3:
        return points.copy(), np.eye(3)

    # PCA → 主轴方向
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # 按特征值降序: 主轴 → 次轴 → 辅轴
    order = np.argsort(eigenvalues)[::-1]
    axes = eigenvectors[:, order]  # (3, 3) 列向量为主轴

    # 确保右手坐标系
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1

    # 对齐: 将主轴旋转到世界坐标轴
    # aligned = centered @ axes → axes 的列就是新的基
    aligned = centered @ axes

    logger.info(
        f"主方向对齐完成: "
        f"特征值比值 λ1:λ2:λ3 = "
        f"{eigenvalues[order][0]:.2f}:{eigenvalues[order][1]:.2f}:{eigenvalues[order][2]:.2f}"
    )

    return aligned, axes


def cluster_semantic_points(
    points: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    class_id_to_name: Dict[int, str],
    eps: float = 0.3,
    min_samples: int = 20,
    min_confidence: float = 0.0,
    target_classes: Optional[List[int]] = None
) -> Tuple[List[OBB], np.ndarray]:
    """
    对语义点云按类分别做 DBSCAN 聚类，每簇拟合 OBB。

    参数:
        points:            (N, 3)
        labels:            (N,) 语义标签
        confidences:       (N,) 标签置信度
        class_id_to_name:  {ADE20K_id: name}
        eps:               DBSCAN eps
        min_samples:       DBSCAN min_samples
        min_confidence:    过滤低置信度点 (0 表示不过滤)
        target_classes:    只处理这些类别 (None = 全部)

    返回:
        instances:   所有 OBB 实例列表
        instance_ids: (N,) 每个点所属实例 (-1 = 噪声)
    """
    valid_mask = labels >= 0
    if min_confidence > 0:
        valid_mask &= confidences >= min_confidence

    unique_classes = np.unique(labels[valid_mask])
    if target_classes is not None:
        unique_classes = [c for c in unique_classes if c in target_classes]

    logger.info(f"开始实例分割: {len(unique_classes)} 个语义类别")

    instances = []
    instance_ids = np.full(len(points), -1, dtype=np.int32)
    next_instance_id = 0

    for class_id in unique_classes:
        class_mask = valid_mask & (labels == class_id)
        class_points = points[class_mask]
        class_indices = np.where(class_mask)[0]

        if len(class_points) < min_samples:
            continue

        class_name = class_id_to_name.get(class_id, f'class_{class_id}')

        # DBSCAN 聚类
        try:
            clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(class_points)
            cluster_labels = clustering.labels_
        except Exception as e:
            logger.warning(f"类别 {class_id} ({class_name}) DBSCAN 失败: {e}")
            continue

        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)

        for cluster_id in range(-1, cluster_labels.max() + 1):
            if cluster_id == -1:
                continue

            cluster_mask = cluster_labels == cluster_id
            cluster_pts = class_points[cluster_mask]
            cluster_orig_indices = class_indices[cluster_mask]

            if len(cluster_pts) < min_samples:
                continue

            try:
                obb = compute_obb(cluster_pts)
                obb.class_id = int(class_id)
                obb.instance_id = next_instance_id
                obb.class_name = class_name
                obb.num_points = len(cluster_pts)

                instances.append(obb)
                instance_ids[cluster_orig_indices] = next_instance_id
                next_instance_id += 1
            except Exception as e:
                logger.debug(f"OBB 拟合失败 (class={class_id}, cluster={cluster_id}): {e}")

    # 统计
    total_points_assigned = (instance_ids >= 0).sum()
    logger.info(
        f"实例分割完成: {len(instances)} 个实例, "
        f"{total_points_assigned} 个点分配 ({total_points_assigned/len(points)*100:.1f}%)"
    )

    # 按类别统计
    class_counts = {}
    for inst in instances:
        class_counts[inst.class_name] = class_counts.get(inst.class_name, 0) + 1
    for cls_name, count in sorted(class_counts.items()):
        logger.info(f"  {cls_name}: {count} 个实例")

    return instances, instance_ids
