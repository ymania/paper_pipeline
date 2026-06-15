#!/usr/bin/env python3
"""
Step 4: 几何引导多视图加权投票 — 论文 §3.4.3-3.4.6 (核心贡献)

算法:
  1. PCA 法向量估计 (kNN, 协方差分解)
  2. 对每个 3D 点，遍历其可见视图:
     - 从 SegFormer 分割图中获取该像素的语义标签
     - 计算距离权重 w_dist = 1/(1+d) 和角度权重 w_angle = |n·v|
     - 综合权重 w = α·w_dist + β·w_angle
     - 累积投票 V_i[l] += w
  3. argmax 选标签，计算置信度

输出:
  - intermediate/semantic_labels.npy    每个点的 ADE20K 标签
  - intermediate/label_confidences.npy  置信度
"""
import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.normal_estimation import estimate_normals
from src.weighted_voting import weighted_vote

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['b0', 'b5'], default='b0',
                        help='使用哪个模型的语义分割结果')
    args_extra = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent

    with open(project_root / 'config' / 'pipeline_params.json', 'r') as f:
        params = json.load(f)

    output_dir = Path('output') if not (project_root / 'config' / 'paths.json').exists() else \
        Path(json.load(open(project_root / 'config' / 'paths.json'))['output_root'])
    intermediate = output_dir / 'intermediate'

    # 加载 step1-3 的中间数据
    points = np.load(intermediate / 'sparse_points.npy')
    logger.info(f"加载点云: {points.shape}")

    with open(intermediate / 'visibility.pkl', 'rb') as f:
        visibility = pickle.load(f)

    with open(intermediate / 'cameras.pkl', 'rb') as f:
        cameras = pickle.load(f)

    with open(intermediate / 'images.pkl', 'rb') as f:
        images = pickle.load(f)

    # 加载语义分割结果
    semantic_dir = intermediate / f'semantic_maps_{args_extra.model}'
    semantic_maps = []
    image_id_to_index = {}

    for i, img_id in enumerate(sorted(images.keys())):
        seg_path = semantic_dir / f'{img_id}.npy'
        if seg_path.exists():
            semantic_maps.append(np.load(seg_path, allow_pickle=True).astype(np.uint8))
            image_id_to_index[img_id] = len(semantic_maps) - 1

    logger.info(f"加载 {len(semantic_maps)} 张语义分割图")

    # Step 4a: 法向量估计 (§3.4.3)
    k = params.get('voting', {}).get('normal_k', 10)
    normals = estimate_normals(points, k=k)

    # Step 4b: 加权投票 (§3.4.4)
    alpha = params.get('voting', {}).get('distance_alpha', 0.5)
    beta = 1.0 - alpha

    logger.info(f"加权投票参数: α={alpha}, β={beta}, k={k}")

    labels, confidences = weighted_vote(
        points=points,
        normals=normals,
        visibility=visibility,
        cameras=cameras,
        images=images,
        semantic_maps=semantic_maps,
        image_id_to_index=image_id_to_index,
        alpha=alpha,
        beta=beta
    )

    # 保存结果
    np.save(intermediate / 'semantic_labels.npy', labels)
    np.save(intermediate / 'label_confidences.npy', confidences)
    np.save(intermediate / 'normals.npy', normals)

    # 统计
    valid = labels >= 0
    n_valid = valid.sum()
    unique_classes = len(np.unique(labels[valid]))

    logger.info(f"加权投票结果: {n_valid}/{len(points)} 点有标签")
    logger.info(f"唯一类别数: {unique_classes}")
    logger.info(f"平均置信度: {confidences[valid].mean():.4f}")
    logger.info(f"标签一致性 (置信度>0.5): {(confidences[valid] > 0.5).sum() / n_valid * 100:.1f}%")

    return 0


if __name__ == '__main__':
    sys.exit(main())
