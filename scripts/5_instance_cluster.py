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
from src.clustering import cluster_semantic_points

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

    logger.info(f"DBSCAN 参数: eps={eps}, min_samples={min_samples}")

    instances, instance_ids = cluster_semantic_points(
        points=points,
        labels=labels,
        confidences=confidences,
        class_id_to_name=class_id_to_name,
        eps=eps,
        min_samples=min_samples
    )

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
