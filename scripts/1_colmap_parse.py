#!/usr/bin/env python3
"""
Step 1: COLMAP 数据解析 — 论文 §3.2
读取 COLMAP sparse/0/ 下的 cameras/images/points3D 文本文件，
保存为 numpy 数组供后续步骤使用。
"""
import sys
import json
import logging
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.colmap_io import read_cameras_txt, read_images_txt, read_points3D_txt

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    config_path = Path(__file__).parent.parent / 'config' / 'paths.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = Path(config['data_root'])
    colmap_dir = data_dir / config['colmap_output']

    # 验证文件存在
    for name in ['cameras.txt', 'images.txt', 'points3D.txt']:
        fpath = colmap_dir / name
        if not fpath.exists():
            raise FileNotFoundError(f"COLMAP 文件不存在: {fpath}")

    # 解析
    cameras = read_cameras_txt(colmap_dir / 'cameras.txt')
    images = read_images_txt(colmap_dir / 'images.txt')
    points, colors, tracks = read_points3D_txt(colmap_dir / 'points3D.txt')

    # 保存
    output_dir = Path(config['output_root']) / 'intermediate'
    output_dir.mkdir(parents=True, exist_ok=True)

    import pickle
    np.save(output_dir / 'points3D.npy', points)
    np.save(output_dir / 'colors.npy', colors)

    # dict 型数据用 pickle
    with open(output_dir / 'cameras.pkl', 'wb') as f:
        pickle.dump(cameras, f)
    with open(output_dir / 'images.pkl', 'wb') as f:
        pickle.dump(images, f)
    with open(output_dir / 'tracks.pkl', 'wb') as f:
        pickle.dump(tracks, f)

    # 统计
    logger.info(f"解析完成: {len(cameras)} 相机, {len(images)} 图像, {len(points)} 3D点")
    logger.info(f"点云范围 X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
    logger.info(f"点云范围 Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
    logger.info(f"点云范围 Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    return 0


if __name__ == '__main__':
    sys.exit(main())
