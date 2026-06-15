#!/usr/bin/env python3
"""
Step 8: Per-Class 评估指标 — 论文 §4.7 改进

生成按语义类别的评估统计，解决论文 §5.2.3 中指出的
"缺乏定量精度评价指标"的局限。

评估维度:
  1. Per-class 语义点统计 (点数、占比、置信度)
  2. Per-class 实例分割效果 (实例数、平均点数、OBB 尺寸)
  3. 建筑构件质量评级 (置信度加权综合评分)

输出:
  - output/evaluation.json     评估报告
  - output/evaluation_table.md  Markdown 表格 (可直接粘贴到论文)
"""

import sys
import json
import logging
import pickle
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 建筑构件关注列表 — 与论文 §3.6.2 映射规则一致
BUILDING_CLASSES = ['wall', 'building', 'windowpane', 'window', 'door',
                    'floor', 'ceiling', 'roof', 'column', 'stairs', 'stairway']


def compute_obb_dims(points: np.ndarray) -> dict:
    """PCA-OBB 尺寸统计 (复用 §3.5.4 公式)"""
    if len(points) < 3:
        return {'length': 0, 'width': 0, 'height': 0, 'volume': 0}
    centered = points - points.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    axes = eigenvectors[:, order]
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    transformed = centered @ axes
    dims = transformed.max(axis=0) - transformed.min(axis=0)
    return {
        'length': float(dims[0]),
        'width': float(dims[1]),
        'height': float(dims[2]),
        'volume': float(np.prod(dims)),
    }


def compute_quality_score(confidence: float, n_points: int, n_instances: int) -> str:
    """综合质量评分 — 基于置信度 + 点数 + 实例数"""
    score = 0
    score += min(confidence * 40, 40)  # 置信度贡献 40%
    score += min(np.log1p(n_points) / np.log(10) * 30, 30)  # 点数贡献 30%
    score += min(n_instances * 3, 30)  # 实例数贡献 30%

    if score >= 80:
        return '★★★★★ 优秀'
    elif score >= 60:
        return '★★★★ 良好'
    elif score >= 40:
        return '★★★ 一般'
    elif score >= 20:
        return '★★ 较差'
    else:
        return '★ 缺失'


def main():
    project_root = Path(__file__).resolve().parent.parent

    output_dir = Path('output') if not (project_root / 'config' / 'paths.json').exists() else \
        Path(json.load(open(project_root / 'config' / 'paths.json'))['output_root'])
    intermediate = output_dir / 'intermediate'

    # 加载数据
    points = np.load(intermediate / 'sparse_points.npy')
    labels = np.load(intermediate / 'semantic_labels.npy')
    confidences = np.load(intermediate / 'label_confidences.npy')
    instance_ids = np.load(intermediate / 'instance_ids.npy')

    with open(intermediate / 'class_mapping.json', 'r') as f:
        class_id_to_name = {int(k): v for k, v in json.load(f).items()}

    try:
        with open(intermediate / 'instances.pkl', 'rb') as f:
            instances = pickle.load(f)
    except FileNotFoundError:
        instances = []

    # === 1. Per-class 语义分割统计 ===
    valid = labels >= 0
    unique_classes = np.unique(labels[valid])
    n_total = len(points)

    semantic_stats = []
    for c in sorted(unique_classes):
        name = class_id_to_name.get(int(c), f'class_{c}')
        mask = labels == c
        n = mask.sum()
        cls_conf = confidences[mask]
        semantic_stats.append({
            'class': name,
            'ade20k_id': int(c),
            'n_points': int(n),
            'ratio_pct': round(n / n_total * 100, 2),
            'mean_conf': round(float(cls_conf.mean()), 3),
            'high_conf_pct': round(float((cls_conf > 0.5).sum() / len(cls_conf) * 100), 1),
            'is_building': name in BUILDING_CLASSES,
        })

    # 按点数排序
    semantic_stats.sort(key=lambda x: x['n_points'], reverse=True)

    # === 2. Per-class 实例分割统计 ===
    instance_stats = []
    instance_classes = set()
    for inst in instances:
        instance_classes.add(inst.class_name)

    for c in sorted(unique_classes):
        name = class_id_to_name.get(int(c), f'class_{c}')
        mask = (labels == c) & (instance_ids >= 0)
        n_assigned = mask.sum()

        # 全局实例统计
        cls_instances = [inst for inst in instances if inst.class_name == name]
        n_inst = len(cls_instances)

        # OBB 尺寸统计
        obb_dims_list = []
        for inst in cls_instances:
            obb_dims_list.append(inst.dimensions)

        if obb_dims_list:
            dims_arr = np.array(obb_dims_list)
            avg_dims = dims_arr.mean(axis=0)
            avg_vol = np.mean([inst.volume for inst in cls_instances])
        else:
            avg_dims = np.array([0, 0, 0])
            avg_vol = 0

        # DBSCAN 噪声率 (未被分配到任何实例的点)
        total_cls = (labels == c).sum()
        noise = total_cls - n_assigned if total_cls > 0 else 0
        noise_rate = noise / total_cls * 100 if total_cls > 0 else 0

        # 综合质量评分
        cls_conf_arr = confidences[labels == c]
        mean_conf = float(cls_conf_arr.mean()) if len(cls_conf_arr) > 0 else 0
        quality = compute_quality_score(mean_conf, n_assigned, n_inst)

        instance_stats.append({
            'class': name,
            'n_instances': n_inst,
            'n_points_assigned': int(n_assigned),
            'noise_rate_pct': round(noise_rate, 1),
            'avg_points_per_instance': round(n_assigned / n_inst, 1) if n_inst > 0 else 0,
            'avg_obb_length': round(float(avg_dims[0]), 3),
            'avg_obb_width': round(float(avg_dims[1]), 3),
            'avg_obb_height': round(float(avg_dims[2]), 3),
            'avg_obb_volume': round(float(avg_vol), 6),
            'quality': quality,
            'is_building': name in BUILDING_CLASSES,
        })

    # === 3. 汇总报告 ===
    building_stats = [s for s in instance_stats if s['is_building'] and s['n_instances'] > 0]
    all_stats = [s for s in instance_stats if s['n_instances'] > 0]

    report = {
        'summary': {
            'total_3d_points': int(n_total),
            'semantic_classes_detected': int(len(unique_classes)),
            'total_instances': len(instances),
            'building_component_classes': len(building_stats),
            'building_component_instances': sum(s['n_instances'] for s in building_stats),
        },
        'semantic_stats': semantic_stats,
        'instance_stats': instance_stats,
        'building_component_quality': building_stats,
        'full_quality': all_stats,
    }

    eval_path = output_dir / 'evaluation.json'
    with open(eval_path, 'w') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"评估报告 → {eval_path}")

    # === 4. 生成 Markdown 表格 ===
    md_lines = [
        "# 语义实例分割评估报告",
        "",
        f"**总点数**: {n_total:,}  |  **语义类别**: {len(unique_classes)}  |  **构件实例**: {len(instances)}",
        "",
        "## 建筑构件实例统计 (§4.5)",
        "",
        "| 构件类别 | 实例数 | 分配点数 | 噪声率 | 平均OBB长(m) | 平均OBB宽(m) | 平均OBB高(m) | 综合质量 |",
        "|----------|--------|----------|--------|-------------|-------------|-------------|----------|",
    ]

    for s in instance_stats:
        if s['is_building'] and s['n_instances'] > 0:
            md_lines.append(
                f"| {s['class']} | {s['n_instances']} | {s['n_points_assigned']} | "
                f"{s['noise_rate_pct']:.1f}% | {s['avg_obb_length']:.3f} | "
                f"{s['avg_obb_width']:.3f} | {s['avg_obb_height']:.3f} | "
                f"{s['quality']} |"
            )

    md_lines.extend([
        "",
        "## 语义类别置信度统计 (§4.4)",
        "",
        "| 类别 | 点数 | 占比 | 平均置信度 | 高置信占比 |",
        "|------|------|------|-----------|-----------|",
    ])

    for s in semantic_stats[:20]:
        md_lines.append(
            f"| {s['class']} | {s['n_points']} | {s['ratio_pct']}% | "
            f"{s['mean_conf']:.3f} | {s['high_conf_pct']}% |"
        )

    md_path = output_dir / 'evaluation_table.md'
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(md_lines))

    # 终端输出
    logger.info("\n" + "=" * 70)
    logger.info("建筑构件实例评估")
    logger.info("=" * 70)
    for s in building_stats:
        logger.info(
            f"  {s['class']:15s} | {s['n_instances']:3d} 实例 | "
            f"{s['n_points_assigned']:5d} 点 | "
            f"OBB {s['avg_obb_length']:.2f}×{s['avg_obb_width']:.2f}×{s['avg_obb_height']:.2f} | "
            f"噪声 {s['noise_rate_pct']:.1f}% | {s['quality']}"
        )

    logger.info(f"\nMarkdown 表格 → {md_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
