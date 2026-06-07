#!/usr/bin/env python3
"""
Step 3: 3D→2D 投影与可见性计算 — 论文 §3.4.1-3.4.2

对每个 COLMAP 稀疏 3D 点，利用 track 信息确定它在哪些图像中可见，
计算投影像素坐标和深度。输出:
  - intermediate/visibility.pkl   每点的可见视图+像素坐标
  - intermediate/sparse_points.npy   (备份)
"""
import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colmap_io import read_cameras_txt, read_images_txt, read_points3D_txt
from src.projection import compute_visibility

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent.parent / 'config' / 'paths.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = Path(config['data_root'])
    colmap_dir = data_dir / config['colmap_output']
    output_dir = Path(config['output_root'])
    intermediate = output_dir / 'intermediate'
    intermediate.mkdir(parents=True, exist_ok=True)

    # 直接从 COLMAP 读取 (不依赖 step1 的输出)
    cameras = read_cameras_txt(colmap_dir / 'cameras.txt')
    images = read_images_txt(colmap_dir / 'images.txt')
    points, colors, tracks = read_points3D_txt(colmap_dir / 'points3D.txt')

    # 构建 point_index —— COLMAP point3d_id → 数组索引
    # tracks 的 key 是 COLMAP 的 point3d_id (从 points3D.txt 读取)
    point_index_map = {}
    for i, p3d_id in enumerate(tracks.keys()):
        point_index_map[p3d_id] = i

    logger.info(f"开始投影: {len(points)} 点, {len(images)} 张图像")

    # 计算可见性
    visibility = compute_visibility(
        points, cameras, images, tracks, point_index_map)

    # 保存
    with open(intermediate / 'visibility.pkl', 'wb') as f:
        pickle.dump(visibility, f)

    np.save(intermediate / 'sparse_points.npy', points)
    np.save(intermediate / 'sparse_colors.npy', colors)

    # 保存 cameras (包含 K 矩阵) 
    with open(intermediate / 'cameras.pkl', 'wb') as f:
        pickle.dump(cameras, f)

    logger.info(f"可见性数据: {len(visibility)} 个点有投影 → {intermediate / 'visibility.pkl'}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
