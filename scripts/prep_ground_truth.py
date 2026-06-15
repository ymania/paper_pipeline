#!/usr/bin/env python3
"""
Step 2a: 生成标注模板 — 采样200个语义点，投影回原图，裁剪图像块供人工标注

输出:
  output/ground_truth/patches/    图像块 (128×128 PNG)
  output/ground_truth/template.json  标注模板
"""
import sys, json, pickle, logging, random
import numpy as np
from pathlib import Path
from PIL import Image

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
SRC_IMAGES = Path('/home/qmyyc/my_workplace/语义管道/project/data/images')
INTERMEDIATE = Path('output/intermediate')
OUT_DIR = Path('output/ground_truth')
N_SAMPLES = 200
PATCH_SIZE = 128
CLASSES = ['wall', 'building', 'window', 'door', 'floor', 'plant', 'sky', 'other']

# 加载数据
points = np.load(INTERMEDIATE / 'sparse_points.npy')
labels = np.load(INTERMEDIATE / 'semantic_labels.npy')
with open(INTERMEDIATE / 'visibility.pkl', 'rb') as f:
    visibility = pickle.load(f)
with open(INTERMEDIATE / 'images.pkl', 'rb') as f:
    images = pickle.load(f)
with open(INTERMEDIATE / 'class_mapping.json', 'r') as f:
    class_id_to_name = {int(k): v for k, v in json.load(f).items()}

# 只采样有标签的点
valid_mask = labels >= 0
valid_indices = np.where(valid_mask)[0]
logger.info(f"有标签点数: {len(valid_indices)}")

if len(valid_indices) < N_SAMPLES:
    sampled = valid_indices
else:
    np.random.seed(42)
    sampled = np.random.choice(valid_indices, N_SAMPLES, replace=False)

logger.info(f"采样: {len(sampled)} 点")

# 创建输出目录
patch_dir = OUT_DIR / 'patches'
patch_dir.mkdir(parents=True, exist_ok=True)

template = {'samples': [], 'classes': CLASSES, 'class_description': {
    'wall': '墙体/墙面',
    'building': '建筑主体(无法细分时)',
    'window': '窗户',
    'door': '门',
    'floor': '地面/地板',
    'plant': '植物/树木',
    'sky': '天空',
    'other': '其他/不确定',
}}

# 生成标注样本
for i, pt_idx in enumerate(sampled):
    vis_info = visibility[pt_idx]
    cam_indices = vis_info['cam_indices']
    px_coords = vis_info['px_coords']

    if not cam_indices:
        continue

    # 取第一个可见视图
    img_id = cam_indices[0]
    u, v = px_coords[0]
    ui, vi = int(round(u)), int(round(v))

    # 找到对应的图像文件
    img_name = images[img_id].get('name', f'image_{img_id}')
    # 在 SRC_IMAGES 中找匹配的 JPG
    img_path = None
    for ext in ['.JPG', '.jpg', '.jpeg', '.png']:
        candidate = SRC_IMAGES / (img_name + ext)
        if candidate.exists():
            img_path = candidate
            break
    if img_path is None:
        # 尝试直接用 img_name
        candidate = SRC_IMAGES / img_name
        if candidate.exists():
            img_path = candidate

    if img_path is None:
        logger.warning(f"  找不到图像: {img_name}")
        continue

    # 读取原图并裁剪
    try:
        img = Image.open(img_path)
        W, H = img.size

        # 裁剪 128×128 以投影点为中心
        x1 = max(0, ui - PATCH_SIZE // 2)
        y1 = max(0, vi - PATCH_SIZE // 2)
        x2 = min(W, x1 + PATCH_SIZE)
        y2 = min(H, y1 + PATCH_SIZE)
        x1 = max(0, x2 - PATCH_SIZE)
        y1 = max(0, y2 - PATCH_SIZE)

        patch = img.crop((x1, y1, x2, y2))
        patch_name = f'sample_{i:04d}.png'
        patch.save(patch_dir / patch_name)

        # 当前 SegFormer 预测的标签
        seg_label = labels[pt_idx]
        seg_class_name = class_id_to_name.get(int(seg_label), f'class_{seg_label}')

        template['samples'].append({
            'id': i,
            'pt_idx': int(pt_idx),
            'patch_file': patch_name,
            'image_name': img_name,
            'px_u': ui,
            'px_v': vi,
            'img_width': W,
            'img_height': H,
            'seg_prediction': seg_class_name,
            'seg_class_id': int(seg_label),
            'ground_truth': '',  # 待填写
        })

    except Exception as e:
        logger.warning(f"  处理点 {pt_idx} 失败: {e}")

# 保存模板
template['total'] = len(template['samples'])
with open(OUT_DIR / 'template.json', 'w') as f:
    json.dump(template, f, indent=2, ensure_ascii=False)

# 打印统计
pred_dist = {}
for s in template['samples']:
    pred = s['seg_prediction']
    pred_dist[pred] = pred_dist.get(pred, 0) + 1

logger.info(f"\n生成 {len(template['samples'])} 个标注样本")
logger.info(f"图像块 → {patch_dir}/")
logger.info(f"模板   → {OUT_DIR / 'template.json'}")
logger.info(f"\nSegFormer 预测分布:")
for cls_name, count in sorted(pred_dist.items(), key=lambda x: -x[1]):
    logger.info(f"  {cls_name:15s}: {count:4d}")

logger.info(f"\n标注指南:")
logger.info(f"  1. 打开 {OUT_DIR / 'template.json'}")
logger.info(f"  2. 查看 output/ground_truth/patches/ 中的图像块")
logger.info(f"  3. 对每个 sample 填写 ground_truth 字段 ({', '.join(CLASSES)})")
logger.info(f"  4. 保存后运行 scripts/eval_ground_truth.py 自动算 IoU")
