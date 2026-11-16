"""
COLMAP 文本格式 I/O — 论文 §3.2 三维重建模块

COLMAP 输出格式 (sparse/0/):
  cameras.txt:  CAMERA_ID, MODEL, WIDTH, HEIGHT, fx, fy, cx, cy, ...
  images.txt:   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
                (每个相机后跟一行 POINTS2D[]，每点格式: X Y POINT3D_ID)
  points3D.txt: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[...]
                TRACK 每项: IMAGE_ID POINT2D_IDX
"""
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """四元数 → 旋转矩阵 (COLMAP 使用 w,x,y,z 顺序)"""
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy**2 - 2*qz**2,     2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [    2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2,     2*qy*qz - 2*qx*qw],
        [    2*qx*qz - 2*qy*qw,     2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2],
    ])


def read_cameras_txt(path: str) -> Dict[int, dict]:
    """
    解析 cameras.txt，返回 {camera_id: {model, width, height, fx, fy, cx, cy}}
    
    COLMAP PINHOLE 模型: params = [fx, fy, cx, cy]
    COLMAP SIMPLE_RADIAL: params = [f, cx, cy, k1]
    """
    cameras = {}
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            cam_id = int(parts[0])
            model = parts[1]
            width = int(parts[2])
            height = int(parts[3])
            params = list(map(float, parts[4:]))

            counts = {'PINHOLE': 4, 'SIMPLE_PINHOLE': 3,
                      'SIMPLE_RADIAL': 4, 'RADIAL': 5, 'OPENCV': 8}
            if model not in counts:
                raise ValueError(f"Unsupported camera model: {model}")
            if len(params) != counts[model]:
                raise ValueError(f"Camera {cam_id}: invalid {model} parameter count")
            if model in ('PINHOLE', 'OPENCV'):
                fx, fy, cx, cy = params[:4]
            else:
                fx = fy = params[0]
                cx, cy = params[1:3]

            cameras[cam_id] = {
                'model': model, 'params': params, 'width': width, 'height': height,
                'fx': fx, 'fy': fy, 'cx': cx, 'cy': cy,
                'K': np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            }

    logger.info(f"加载 {len(cameras)} 个相机")
    return cameras


def read_images_txt(path: str) -> Dict[int, dict]:
    """
    解析 images.txt，返回 {image_id: {qw,qx,qy,qz, tx,ty,tz, camera_id, name, points2d}}
    
    points2d: list of (x, y, point3d_id)，即该图像上观测到的二维特征点及其对应的 3D 点
    """
    images = {}
    with open(path, 'r') as f:
        lines = [l.strip() for l in f if not l.lstrip().startswith('#')]
    
    i = 0
    while i < len(lines):
        if not lines[i]:
            i += 1
            continue
        parts = lines[i].split()
        img_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        cam_id = int(parts[8])
        name = parts[9]

        # 下一行是 points2D
        points2d = []
        points2d_all = []
        if i + 1 < len(lines):
            pts = lines[i + 1].split()
            for j in range(0, len(pts), 3):
                x = float(pts[j])
                y = float(pts[j + 1])
                p3d_id = int(pts[j + 2])
                points2d_all.append((x, y, p3d_id))
                if p3d_id >= 0:
                    points2d.append((x, y, p3d_id))

        R = qvec2rotmat(np.array([qw, qx, qy, qz]))
        t = np.array([tx, ty, tz])
        c = -R.T @ t  # 相机中心在世界坐标系下的位置

        images[img_id] = {
            'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
            'tx': tx, 'ty': ty, 'tz': tz,
            'camera_id': cam_id,
            'name': name,
            'R': R, 't': t, 'C': c,
            'points2d': points2d, 'points2d_all': points2d_all
        }
        i += 2

    logger.info(f"加载 {len(images)} 张图像")
    return images


def read_points3D_txt(path: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, list]]:
    """
    解析 points3D.txt，返回 (points, colors, tracks)
    
    points: (N, 3) float32
    colors: (N, 3) uint8 (0-255)
    tracks: {point3d_id: [(image_id, point2d_idx), ...]}
    """
    points_list = []
    colors_list = []
    tracks = {}
    point_index = {}  # COLMAP point3d_id -> array index
    idx = 0

    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            p3d_id = int(parts[0])
            x, y, z = map(float, parts[1:4])
            r, g, b = map(int, parts[4:7])
            error = float(parts[7])
            
            # TRACK: (image_id, point2d_idx) pairs
            track = []
            for j in range(8, len(parts), 2):
                img_id = int(parts[j])
                pt2d_idx = int(parts[j + 1])
                track.append((img_id, pt2d_idx))

            points_list.append([x, y, z])
            colors_list.append([r, g, b])
            tracks[p3d_id] = track
            point_index[p3d_id] = idx
            idx += 1

    points = np.array(points_list, dtype=np.float32)
    colors = np.array(colors_list, dtype=np.uint8)

    logger.info(f"加载 {len(points)} 个 3D 点")
    return points, colors, tracks


def get_camera_matrix(cam: dict) -> np.ndarray:
    """返回 3×3 内参矩阵 K"""
    return cam['K']


def build_view_visibility(images: dict, tracks: dict) -> Dict[int, list]:
    """
    从 COLMAP tracks 构建每个点的可见视图列表。
    
    返回: {point_array_index: [image_id, ...]}
    注意: key 是点云数组中的索引(0-based)，不是 COLMAP 的 point3d_id
    """
    # 先建 COLMAP point3d_id -> array index 的映射
    # tracks 的 key 是 COLMAP 的 point3d_id
    # 我们需要反查: 每个 COLMAP point3d_id 对应哪个数组索引
    # 这里直接用 tracks 的 key 存对应关系
    
    # 重建 point_index
    point_index = {p3d_id: i for i, p3d_id in enumerate(tracks.keys())}
    
    visibility = {}
    for p3d_id, track in tracks.items():
        idx = point_index.get(p3d_id, p3d_id)  # fallback if mapping differs
        vis_images = list(dict.fromkeys(img_id for img_id, _ in track))
        visibility[idx] = vis_images
    
    # 确保 COLMAP point3d_id 和数组索引对齐
    # 如果不对齐，直接用 enumerate
    if len(visibility) != len(tracks):
        visibility = {}
        for i, (p3d_id, track) in enumerate(tracks.items()):
            visibility[i] = [img_id for img_id, _ in track]
    
    return visibility
