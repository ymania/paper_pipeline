"""
3D→2D 投影与可见性计算 — 论文 §3.4.1-3.4.2

投影公式 (§3.4.2):
  [u, v, 1]^T = K · [R|t] · [p, 1]^T

(已齐次化，除以 z 得到像素坐标)
"""
import numpy as np
import logging
from typing import Optional, Tuple, List

logger = logging.getLogger(__name__)


def project_point_to_camera(
    point3d: np.ndarray,
    cam: dict,
    img: dict
) -> Optional[Tuple[float, float, float]]:
    """
    将一个 3D 点投影到指定相机。

    参数:
        point3d:  [x, y, z] 世界坐标
        cam:      camera dict (含 K, width, height)
        img:      image dict (含 R, t)

    返回:
        (u, v, depth) 像素坐标和深度；若在相机后方或图像外返回 None
    """
    K = cam['K']
    R = img['R']
    t = img['t']

    # 世界坐标 → 相机坐标
    p_cam = R @ point3d + t
    
    if p_cam[2] <= 0:
        return None  # 在相机后方

    # 相机坐标 → 像素坐标
    p_img = K @ p_cam
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]
    depth = p_cam[2]

    # 边界检查 (margin=5)
    margin = 5
    if (u < -margin or u >= cam['width'] + margin or
        v < -margin or v >= cam['height'] + margin):
        return None

    return float(u), float(v), float(depth)


def compute_visibility(
    points: np.ndarray,
    cameras: dict,
    images: dict,
    tracks: dict,
    point_index_map: dict = None
) -> dict:
    """
    计算所有 3D 点在所有相机下的投影可见性。

    利用 COLMAP tracks 信息加速: 只投影该点被实际观测到的图像。

    返回:
        {
            point_idx: {
                'cam_indices': [img_id, ...],
                'px_coords': [(u, v), ...],
                'depths': [depth, ...]
            }
        }
    """
    N = len(points)
    visibility = {}

    # 从 tracks 获取每个点被哪些图像观测到
    for p3d_id, track in tracks.items():
        # p3d_id 在 tracks 中是 COLMAP 的 point3d_id
        # 需要映射到数组索引
        if point_index_map:
            idx = point_index_map.get(p3d_id)
        else:
            idx = p3d_id  # 假设已对齐
        
        if idx is None or idx >= N:
            continue

        vis_info = {
            'cam_indices': [],
            'px_coords': [],
            'depths': []
        }

        pt = points[idx]
        for img_id, _ in track:
            if img_id not in images or img_id not in cameras:
                continue
            
            img = images[img_id]
            cam = cameras[img['camera_id']]
            
            result = project_point_to_camera(pt, cam, img)
            if result is not None:
                u, v, depth = result
                vis_info['cam_indices'].append(img_id)
                vis_info['px_coords'].append((u, v))
                vis_info['depths'].append(depth)

        visibility[idx] = vis_info

        if (idx + 1) % 10000 == 0:
            logger.info(f"可见性计算: {idx+1}/{N}")

    # 统计
    vis_counts = [len(v['cam_indices']) for v in visibility.values()]
    if vis_counts:
        logger.info(
            f"可见性统计: 最少={min(vis_counts)}, 最多={max(vis_counts)}, "
            f"平均={np.mean(vis_counts):.2f}, "
            f"有投影点数={len(visibility)}/{N}"
        )

    return visibility


def compute_viewing_direction(camera_center: np.ndarray, point: np.ndarray) -> np.ndarray:
    """
    计算观测方向向量 (从 3D 点指向相机中心，单位向量)。

    参数:
        camera_center: 相机世界坐标 [cx, cy, cz]
        point: 3D 点世界坐标 [px, py, pz]

    返回:
        单位观测方向向量
    """
    direction = camera_center - point
    norm = np.linalg.norm(direction)
    if norm < 1e-10:
        return np.array([0.0, 0.0, 1.0])
    return direction / norm
