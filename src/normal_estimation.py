"""
点云法向量估计 (PCA 方法) — 论文 §3.4.3

对每个 3D 点 p_i，取其 k 近邻，计算协方差矩阵 Σ，
最小特征值对应的特征向量即为法向量 n_i。

协方差矩阵:
  Σ_i = (1/|N_i|) Σ (p - μ_i)(p - μ_i)^T

曼哈顿世界假设: 建筑场景中法向量在单位球上呈 3 个正交聚类。
"""
import numpy as np
import logging
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


def estimate_normals(
    points: np.ndarray,
    k: int = 10
) -> np.ndarray:
    """
    使用 PCA 对稀疏点云计算法向量。

    参数:
        points: (N, 3) 点云坐标
        k:      近邻点数 (默认 10)

    返回:
        normals: (N, 3) 单位法向量 (方向已做一致性处理，指向视点更多的一侧)
    """
    N = len(points)
    if N < k:
        logger.warning(f"点数 ({N}) < k ({k})，使用所有点")
        k = N

    logger.info(f"计算 {N} 个点的法向量 (k={k})...")

    # 构建 kNN 搜索
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, N), algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    normals = np.zeros((N, 3), dtype=np.float64)

    for i in range(N):
        # 获取 k 近邻 (排除自身)
        knn_idx = indices[i, 1:min(k + 1, N)]  # skip self

        if len(knn_idx) < 3:
            normals[i] = [0, 0, 1]
            continue

        neighbors = points[knn_idx]
        centroid = neighbors.mean(axis=0)

        # 协方差矩阵
        centered = neighbors - centroid
        cov = (centered.T @ centered) / len(knn_idx)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # 最小特征值对应的特征向量 = 法向量
        normal = eigenvectors[:, 0]

        # 归一化
        norm = np.linalg.norm(normal)
        if norm > 1e-10:
            normal /= norm

        normals[i] = normal

    # 方向一致性处理：让所有法向量指向大致相同的半球
    # 对建筑场景，Z 分量通常应指向外 (向上或向外)
    mean_normal = normals.mean(axis=0)
    if np.linalg.norm(mean_normal) > 1e-10:
        mean_normal /= np.linalg.norm(mean_normal)
        # 翻转向量与平均方向相反的法向量
        dots = np.dot(normals, mean_normal)
        flip_mask = dots < 0
        normals[flip_mask] *= -1

    logger.info(f"法向量计算完成，平均 Z 分量: {normals[:, 2].mean():.4f}")

    return normals.astype(np.float32)
