#!/usr/bin/env python3
"""
Step 7: 可视化 — 生成 HTML 交互式点云 + 统计图表

输出:
  - output/visualization.html    Plotly 3D 可视化
  - output/stats.json            统计摘要
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


def main():
    project_root = Path(__file__).resolve().parent.parent

    output_dir = Path('output') if not (project_root / 'config' / 'paths.json').exists() else \
        Path(json.load(open(project_root / 'config' / 'paths.json'))['output_root'])
    intermediate = output_dir / 'intermediate'

    # 加载所有数据
    points = np.load(intermediate / 'sparse_points.npy')
    colors_rgb = np.load(intermediate / 'sparse_colors.npy')
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

    # 统计
    valid = labels >= 0
    n_valid = valid.sum()
    n_total = len(points)

    unique_classes = np.unique(labels[valid])
    class_counts = {}
    class_stats = {}  # per-class detailed stats
    for c in unique_classes:
        name = class_id_to_name.get(int(c), f'class_{c}')
        mask = labels == c
        count = mask.sum()
        class_counts[name] = int(count)

        # Per-class metrics
        cls_conf = confidences[mask] if mask.sum() > 0 else np.array([0])
        cls_inst_mask = instance_ids[mask]
        n_inst = len(set(cls_inst_mask[cls_inst_mask >= 0]))

        class_stats[name] = {
            'class_id': int(c),
            'point_count': int(count),
            'point_ratio_pct': round(float(count / n_total * 100), 2) if n_total > 0 else 0.0,
            'mean_confidence': round(float(cls_conf.mean()), 4) if len(cls_conf) > 0 else 0.0,
            'high_conf_ratio_pct': round(float((cls_conf > 0.5).sum() / len(cls_conf) * 100), 2) if len(cls_conf) > 0 else 0.0,
            'num_instances': n_inst,
        }

    # 排序
    class_counts_sorted = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)

    stats = {
        'total_points': int(n_total),
        'labeled_points': int(n_valid),
        'label_coverage': float(n_valid / n_total * 100),
        'unique_classes': int(len(unique_classes)),
        'mean_confidence': float(confidences[valid].mean()) if n_valid > 0 else 0.0,
        'high_confidence_ratio': float((confidences[valid] > 0.5).sum() / n_valid * 100) if n_valid > 0 else 0.0,
        'num_instances': len(instances),
        'class_distribution': dict(class_counts_sorted[:20]),
        'per_class_stats': class_stats,
        'point_cloud_range': {
            'x': [float(points[:, 0].min()), float(points[:, 0].max())],
            'y': [float(points[:, 1].min()), float(points[:, 1].max())],
            'z': [float(points[:, 2].min()), float(points[:, 2].max())],
        }
    }

    with open(output_dir / 'stats.json', 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    logger.info(f"统计摘要 → {output_dir / 'stats.json'}")

    # 打印关键指标 (论文数据)
    logger.info("=" * 50)
    logger.info("论文关键指标:")
    logger.info(f"  总 3D 点数:        {n_total:,}")
    logger.info(f"  有语义标签的点:    {n_valid:,} ({stats['label_coverage']:.1f}%)")
    logger.info(f"  唯一语义类别:      {stats['unique_classes']}")
    logger.info(f"  平均置信度:        {stats['mean_confidence']:.4f}")
    logger.info(f"  高置信度占比:      {stats['high_confidence_ratio']:.1f}%")
    logger.info(f"  构件实例数:        {stats['num_instances']}")
    logger.info("=" * 50)
    logger.info("Top-10 类别:")
    for name, count in class_counts_sorted[:10]:
        cs = class_stats.get(name, {})
        n_inst = cs.get('num_instances', 0)
        conf = cs.get('mean_confidence', 0)
        logger.info(f"  {name:20s} {count:8,d} 点  |  {n_inst:3d} 实例  |  置信度 {conf:.3f}")

    # 生成 HTML 可视化 (Plotly)
    html = generate_plotly_html(points, labels, colors_rgb, class_id_to_name, instances)
    html_path = output_dir / 'visualization.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html)

    logger.info(f"可视化 → {html_path}")

    return 0


def generate_plotly_html(points, labels, colors_rgb, class_id_to_name, instances):
    """生成 Plotly 3D 点云 HTML"""
    
    valid = labels >= 0
    n = min(50000, valid.sum())  # 最多采样 5 万个点用于显示
    
    if n < valid.sum():
        idx = np.random.choice(np.where(valid)[0], n, replace=False)
    else:
        idx = np.where(valid)[0]
    
    pts = points[idx]
    labs = labels[idx]
    
    # 为每个类别分配颜色
    unique_labs = np.unique(labs)
    np.random.seed(42)
    import colorsys
    color_palette = {}
    for i, lab in enumerate(unique_labs):
        hue = i / len(unique_labs)
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.9)
        color_palette[lab] = f'rgb({int(r*255)},{int(g*255)},{int(b*255)})'

    marker_colors = [color_palette.get(l, 'rgb(128,128,128)') for l in labs]
    
    # 构建 hover text
    hover_texts = []
    for l in labs:
        name = class_id_to_name.get(int(l), f'class_{l}')
        hover_texts.append(f'{name} (ID:{l})')

    html = f'''<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>语义点云可视化</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body {{ margin: 0; background: #111; font-family: monospace; }}
  #plot {{ width: 100vw; height: 100vh; }}
</style>
</head>
<body>
<div id="plot"></div>
<script>
var trace = {{
    type: 'scatter3d',
    mode: 'markers',
    x: {pts[:, 0].tolist()},
    y: {pts[:, 1].tolist()},
    z: {pts[:, 2].tolist()},
    text: {json.dumps(hover_texts)},
    marker: {{
        size: 1.5,
        color: {json.dumps(marker_colors)},
        opacity: 0.8
    }}
}};
var layout = {{
    paper_bgcolor: '#111',
    scene: {{
        xaxis: {{ title: 'X', color: '#888', gridcolor: '#222' }},
        yaxis: {{ title: 'Y', color: '#888', gridcolor: '#222' }},
        zaxis: {{ title: 'Z', color: '#888', gridcolor: '#222' }},
    }},
    margin: {{ l:0, r:0, t:30, b:0 }},
    title: {{ text: '稀疏语义点云 (Semantic Sparse Point Cloud)', font: {{ color: '#ccc', size: 16 }} }}
}};
Plotly.newPlot('plot', [trace], layout);
</script>
</body>
</html>'''

    return html


if __name__ == '__main__':
    sys.exit(main())
