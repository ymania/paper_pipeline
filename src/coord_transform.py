"""
坐标转换工具：COLMAP 视觉坐标系 → IFC 建筑坐标系。
导师指导要点：
- COLMAP Z轴朝前（右手系），IFC 需 Z轴朝上
- 需要缩放到米（IFC标准单位）
- 让建筑基点移到 IFC 原点(0,0,0)
"""
import numpy as np
import logging

logger = logging.getLogger(__name__)


def transform_to_ifc_space(colmap_points, scale_factor=1.0, flip_z=True, translation=None):
    """
    COLMAP 坐标 → IFC 坐标转换。

    参数:
        colmap_points: (N, 3) COLMAP坐标系下的点
        scale_factor: 将COLMAP单位缩放到米（IFC标准单位）
            调试技巧：在场景中找一个已知尺寸的物体（如窗户宽1米），
            对比点云中该物体的宽度来计算
        flip_z: COLMAP默认Z轴朝前（右手系），IFC需Z轴朝上
        translation: 手动平移，将建筑基点移到IFC原点(0,0,0)

    返回:
        (N, 3) IFC坐标系下的点
    """
    ifc_points = colmap_points.copy().astype(np.float64)
    ifc_points *= scale_factor  # 缩放

    if flip_z:
        ifc_points[:, 2] *= -1.0  # Z轴翻转

    if translation is None:
        # 自动对齐：Z轴归零 + XY居中
        min_z = np.min(ifc_points[:, 2])
        ifc_points[:, 2] -= min_z  # 让Z >= 0
        center_xy = np.mean(ifc_points[:, :2], axis=0)
        ifc_points[:, :2] -= center_xy  # XY居中
    else:
        ifc_points += np.array(translation)

    return ifc_points


def estimate_scale_factor(colmap_points, known_length_meters, point_indices=None):
    """
    通过已知尺寸物体估算缩放因子。
    例如：已知某窗户宽1.2米，在点云中找到对应两个点，距离为0.8（COLMAP单位），
    则 scale_factor = 1.2 / 0.8 = 1.5

    参数:
        colmap_points: (N, 3) 点云
        known_length_meters: 已知物体的真实长度（米）
        point_indices: 两个端点的索引 [i, j]，如果不提供则使用点云最远两点

    返回:
        scale_factor: float
    """
    if point_indices is not None and len(point_indices) == 2:
        i, j = point_indices
        colmap_length = np.linalg.norm(colmap_points[i] - colmap_points[j])
    else:
        # 用点云包围盒最大维度估算（不太准确，仅供参考）
        bbox_range = np.max(colmap_points, axis=0) - np.min(colmap_points, axis=0)
        colmap_length = np.max(bbox_range)
        logger.warning("未指定参考点，使用包围盒最大维度，缩放因子可能不准确")

    if colmap_length < 1e-10:
        return 1.0

    scale_factor = known_length_meters / colmap_length
    logger.info(f"估算缩放因子: {scale_factor:.4f} (COLMAP长度={colmap_length:.4f}, 真实长度={known_length_meters:.4f}m)")
    return scale_factor
