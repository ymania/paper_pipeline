#!/usr/bin/env python3
"""
Step 5: DBSCAN 实例分割 + OBB 包围盒 — 论文 §3.5

对每个语义类别分别 DBSCAN 聚类，提取独立建筑构件实例，
PCA-OBB 拟合包围盒。

输出:
  - intermediate/instances.pkl         OBB 实例列表
  - intermediate/instance_ids.npy      每个点所属实例
"""
import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.clustering import cluster_semantic_points, filter_outliers_lof, align_to_principal_axes

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parent.parent

    with open(project_root / 'config' / 'pipeline_params.json', 'r') as f:
        params = json.load(f)

    output_dir = Path('output') if not (project_root / 'config' / 'paths.json').exists() else \
        Path(json.load(open(project_root / 'config' / 'paths.json'))['output_root'])
    intermediate = output_dir / 'intermediate'

    # 加载数据
    points = np.load(intermediate / 'sparse_points.npy')
    labels = np.load(intermediate / 'semantic_labels.npy')
    confidences = np.load(intermediate / 'label_confidences.npy')

    with open(intermediate / 'class_mapping.json', 'r') as f:
        class_id_to_name = {int(k): v for k, v in json.load(f).items()}

    # DBSCAN 参数
    eps = params.get('clustering', {}).get('eps', 0.3)
    min_samples = params.get('clustering', {}).get('min_samples', 20)
    use_lof = params.get('clustering', {}).get('use_lof', True)
    lof_contamination = params.get('clustering', {}).get('lof_contamination', 0.1)
    use_alignment = params.get('clustering', {}).get('use_alignment', True)

    logger.info(f"DBSCAN 参数: eps={eps}, min_samples={min_samples}")
    logger.info(f"LOF 预过滤: {'开启' if use_lof else '关闭'} (contamination={lof_contamination})")
    logger.info(f"主方向对齐: {'开启' if use_alignment else '关闭'}")

    # Step 5a: 主方向对齐 — 参考张曾一(2025) §3.3
    rotation = np.eye(3)
    raw_points = points.copy()
    if use_alignment:
        points, rotation = align_to_principal_axes(points)
        np.save(intermediate / 'alignment_rotation.npy', rotation)
        np.save(intermediate / 'aligned_points.npy', points)
        logger.info(f"对齐后点云范围: X=[{points[:,0].min():.2f}, {points[:,0].max():.2f}], "
                    f"Y=[{points[:,1].min():.2f}, {points[:,1].max():.2f}], "
                    f"Z=[{points[:,2].min():.2f}, {points[:,2].max():.2f}]")

    # Step 5b: LOF 离群点预过滤 — 参考张曾一(2025) §3.5, 葛健(2024) §3.2
    if use_lof:
        valid_mask_init = labels >= 0
        points_init = points[valid_mask_init]
        labels_init = labels[valid_mask_init]
        conf_init = confidences[valid_mask_init]
        orig_indices = np.where(valid_mask_init)[0]

        # 对每个语义类分别做 LOF（避免跨类密度差异导致误判）
        unique_cls = np.unique(labels_init)
        all_inlier_masks = []
        total_removed = 0
        
        for cls_id in unique_cls:
            cls_mask = labels_init == cls_id
            cls_pts = points_init[cls_mask]
            if len(cls_pts) < 20:
                all_inlier_masks.append(np.ones(cls_mask.sum(), dtype=bool))
                continue
            _, inlier_mask_i, n_removed_i = filter_outliers_lof(
                cls_pts,
                n_neighbors=min(20, len(cls_pts) - 1),
                contamination=lof_contamination
            )
            all_inlier_masks.append(inlier_mask_i)
            total_removed += n_removed_i

        # 合并过滤结果
        combined_inlier = np.concatenate(all_inlier_masks)
        points = points_init[combined_inlier]
        labels = labels_init[combined_inlier]
        confidences = conf_init[combined_inlier]
        
        logger.info(f"LOF 总计剔除: {total_removed} 个离群点")

    logger.info(f"输入点云: {len(points)} 点 → 准备 DBSCAN 聚类")

    instances, instance_ids_filtered = cluster_semantic_points(
        points=points,
        labels=labels,
        confidences=confidences,
        class_id_to_name=class_id_to_name,
        eps=eps,
        min_samples=min_samples
    )

    # 将 instance_ids 扩展回原始点云大小 (LOF 过滤掉的点标为 -1)
    if use_lof:
        instance_ids = np.full(len(raw_points), -1, dtype=np.int32)
        # combined_inlier 索引定位到原始 valid_mask_init 中的位置
        valid_indices_in_full = orig_indices[combined_inlier]
        instance_ids[valid_indices_in_full] = instance_ids_filtered
    else:
        instance_ids = instance_ids_filtered

    # 保存
    with open(intermediate / 'instances.pkl', 'wb') as f:
        pickle.dump(instances, f)

    np.save(intermediate / 'instance_ids.npy', instance_ids)

    # 导出实例信息 JSON
    instances_json = [inst.to_dict() for inst in instances]
    with open(intermediate / 'instances.json', 'w') as f:
        json.dump(instances_json, f, indent=2, ensure_ascii=False)

    logger.info(f"实例分割完成: {len(instances)} 个构件")

    return 0


if __name__ == '__main__':
    sys.exit(main())
