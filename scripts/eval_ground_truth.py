#!/usr/bin/env python3
"""
Step 2b: 评估标注结果 — 算 per-class IoU + mIoU

用法:
  python scripts/eval_ground_truth.py                    # 从 template.json 读取标注
  python scripts/eval_ground_truth.py --auto             # 先自动用SegFormer标签填充
"""
import sys, json, argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

PROJECT = Path(__file__).resolve().parent.parent
TEMPLATE = PROJECT / 'output' / 'ground_truth' / 'template.json'


def auto_fill():
    """用 SegFormer 预测自动填充 ground_truth（未人工标注前的基线）"""
    with open(TEMPLATE, 'r') as f:
        data = json.load(f)

    for sample in data['samples']:
        if not sample['ground_truth']:
            sample['ground_truth'] = sample['seg_prediction']

    with open(TEMPLATE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"已填充 {data['total']} 个样本 (SegFormer预测 → ground_truth)")


def compute_iou(pred: set, gt: set) -> float:
    inter = len(pred & gt)
    union = len(pred | gt)
    return inter / union if union > 0 else 0.0


def compute_metrics(samples):
    """计算 per-class IoU + 准确率"""
    # 构建 per-class TP/FP/FN
    all_classes = set()
    for s in samples:
        all_classes.add(s['seg_prediction'])
        all_classes.add(s['ground_truth'])

    per_class = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'total_gt': 0, 'total_pred': 0})
    correct = 0
    total = len(samples)

    for s in samples:
        pred = s['seg_prediction']
        gt = s['ground_truth']

        per_class[pred]['total_pred'] += 1
        per_class[gt]['total_gt'] += 1

        if pred == gt:
            correct += 1
            per_class[gt]['tp'] += 1
        else:
            per_class[pred]['fp'] += 1  # 预测了pred但实际是gt
            per_class[gt]['fn'] += 1   # 实际是gt但预测了pred

    # 计算指标
    results = {}
    for cls_name in sorted(all_classes):
        stats = per_class[cls_name]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0

        results[cls_name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'iou': round(iou, 4),
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'total_gt': stats['total_gt'],
            'total_pred': stats['total_pred'],
        }

    # mIoU (仅计算有 ground truth 的类别)
    gt_classes = [c for c in all_classes if per_class[c]['total_gt'] > 0]
    miou = np.mean([results[c]['iou'] for c in gt_classes]) if gt_classes else 0

    return {
        'accuracy': round(correct / total, 4),
        'total_samples': total,
        'mIoU': round(float(miou), 4),
        'per_class': results,
    }


def print_report(metrics):
    print("\n" + "=" * 70)
    print("Ground Truth 评估报告")
    print("=" * 70)
    print(f"  总样本数:   {metrics['total_samples']}")
    print(f"  准确率:     {metrics['accuracy']*100:.1f}%")
    print(f"  mIoU:       {metrics['mIoU']:.4f}")
    print()
    print(f"{'类别':<15} {'GT数':>5} {'Pred数':>5} {'TP':>5} {'FP':>5} {'FN':>5} {'Precision':>9} {'Recall':>7} {'IoU':>7}")
    print("-" * 70)

    for cls_name, stats in sorted(metrics['per_class'].items()):
        if stats['total_gt'] == 0 and stats['total_pred'] == 0:
            continue
        print(f"{cls_name:<15} {stats['total_gt']:>5} {stats['total_pred']:>5} "
              f"{stats['tp']:>5} {stats['fp']:>5} {stats['fn']:>5} "
              f"{stats['precision']:>9.4f} {stats['recall']:>7.4f} {stats['iou']:>7.4f}")

    print()

    # 未完成标注的统计
    unlabeled = sum(1 for s in metrics.get('_samples', []) if s.get('ground_truth', '') == '')
    if unlabeled > 0:
        print(f"⚠ 尚有 {unlabeled} 个样本未标注 (ground_truth 为空)")
    else:
        unedited = sum(1 for s in metrics.get('_samples', []) if s.get('ground_truth') == s.get('seg_prediction'))
        print(f"  标注进度: {metrics['total_samples'] - unedited}/{metrics['total_samples']} 已修改")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--auto', action='store_true', help='自动填充 SegFormer 预测为 ground truth')
    args = parser.parse_args()

    if args.auto:
        auto_fill()
        return

    with open(TEMPLATE, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    metrics = compute_metrics(samples)
    metrics['_samples'] = samples

    print_report(metrics)

    # 保存结果
    out_path = PROJECT / 'output' / 'ground_truth' / 'evaluation.json'
    with open(out_path, 'w') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"评估结果 → {out_path}")


if __name__ == '__main__':
    main()
