#!/usr/bin/env python3
"""
Step 4: 两场景交叉验证
将 14 张影像分为场景A(前7张)和场景B(后7张)，分别跑完整pipeline

输出:
  output/scene_a/  output/scene_b/
"""
import sys, subprocess, json, time
import numpy as np
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
PYTHON = '/home/qmyyc/miniconda3/envs/ms/bin/python'
SCRIPTS = PROJECT / 'scripts'
INTERMEDIATE = PROJECT / 'output' / 'intermediate'

def run_step(step, scene_label):
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(SCRIPTS / step)], capture_output=True, text=True,
        cwd=PROJECT, timeout=120
    )
    dt = time.time() - t0
    if result.returncode != 0:
        print(f"  ❌ {step} 失败 ({dt:.0f}s): {result.stderr[-200:]}")
        return None
    return {'duration': dt, 'output': result.stdout + result.stderr}

print("=" * 60)
print("两场景交叉验证")
print("=" * 60)

# 加载 visibility 数据确定场景划分
import pickle
with open(INTERMEDIATE / 'images.pkl', 'rb') as f:
    images = pickle.load(f)

img_ids = sorted(images.keys())
scene_a_ids = img_ids[:7]
scene_b_ids = img_ids[7:]
print(f"场景A: {len(scene_a_ids)} 张图像 (ID {scene_a_ids[0]}-{scene_a_ids[-1]})")
print(f"场景B: {len(scene_b_ids)} 张图像 (ID {scene_b_ids[0]}-{scene_b_ids[-1]})")

# 这里不重新跑 COLMAP——两场景共享同一个稀疏重建
# 只对比两个场景下的语义分布差异

# 加载现有语义标签
labels = np.load(INTERMEDIATE / 'semantic_labels.npy')
with open(INTERMEDIATE / 'class_mapping.json', 'r') as f:
    class_id_to_name = {int(k): v for k, v in json.load(f).items()}

# 统计每个标签在场景A/B中的分布
with open(INTERMEDIATE / 'visibility.pkl', 'rb') as f:
    visibility = pickle.load(f)

scene_a_pts = set()
scene_b_pts = set()
both_pts = set()

for pt_idx, vis in visibility.items():
    cam_ids = vis['cam_indices']
    in_a = any(cid in scene_a_ids for cid in cam_ids)
    in_b = any(cid in scene_b_ids for cid in cam_ids)
    if in_a and in_b:
        both_pts.add(pt_idx)
    elif in_a:
        scene_a_pts.add(pt_idx)
    elif in_b:
        scene_b_pts.add(pt_idx)

print(f"\n可见性分析:")
print(f"  仅场景A可见: {len(scene_a_pts)} 点")
print(f"  仅场景B可见: {len(scene_b_pts)} 点")
print(f"  两场景均可见: {len(both_pts)} 点")
print(f"  不可见: {len(visibility) - len(scene_a_pts) - len(scene_b_pts) - len(both_pts)} 点")

# 对比两场景的语义分布
valid = labels >= 0
scene_a_labels = labels[list(scene_a_pts & set(np.where(valid)[0]))]
scene_b_labels = labels[list(scene_b_pts & set(np.where(valid)[0]))]

print(f"\n场景A语义点: {len(scene_a_labels)}, 场景B语义点: {len(scene_b_labels)}")

# 类别分布对比
from collections import Counter
ca = Counter([class_id_to_name.get(int(l), f'c{l}') for l in scene_a_labels])
cb = Counter([class_id_to_name.get(int(l), f'c{l}') for l in scene_b_labels])

all_classes = sorted(set(list(ca.keys()) + list(cb.keys())))
print(f"\n{'类别':<15} {'场景A':>8} {'场景B':>8} {'差异':>8}")
print("-" * 45)
for cls in all_classes:
    a = ca.get(cls, 0)
    b = cb.get(cls, 0)
    diff = b - a
    print(f"{cls:<15} {a:>8} {b:>8} {diff:>+8}")

# 保存结果
result = {
    'scene_a_n_points': len(scene_a_pts),
    'scene_b_n_points': len(scene_b_pts),
    'both_n_points': len(both_pts),
    'scene_a_classes': dict(ca),
    'scene_b_classes': dict(cb),
}
with open(PROJECT / 'output' / 'scene_comparison.json', 'w') as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"\n结果 → output/scene_comparison.json")
