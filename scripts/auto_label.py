#!/usr/bin/env python3
"""
自动标注：CLIP 零样本分类 200 个图像块
"""
import sys, json
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parent.parent
PATCH_DIR = PROJECT / 'output' / 'ground_truth' / 'patches'
TEMPLATE = PROJECT / 'output' / 'ground_truth' / 'template.json'
CLASSES = ['wall', 'building facade', 'window', 'door', 'floor or ground',
           'plant or tree', 'sky', 'person', 'other']

def main():
    import torch
    from transformers import CLIPProcessor, CLIPModel

    device = "cpu"
    print(f"加载 CLIP...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model.eval()

    with open(TEMPLATE, 'r') as f:
        data = json.load(f)

    class_texts = [f"a photo of a {c}" for c in CLASSES]
    label_map = {
        'wall': 'wall', 'building facade': 'building',
        'window': 'window', 'door': 'door',
        'floor or ground': 'floor', 'plant or tree': 'plant',
        'sky': 'sky', 'person': 'person', 'other': 'other',
    }

    # 批量加载所有图像
    images = []
    valid_samples = []
    for sample in data['samples']:
        patch_path = PATCH_DIR / sample['patch_file']
        if patch_path.exists():
            try:
                images.append(Image.open(patch_path).convert('RGB'))
                valid_samples.append(sample)
            except Exception:
                pass

    print(f"有效图像: {len(images)}/{len(data['samples'])}")

    # 分批处理 (每批 50 张避免 OOM)
    BATCH = 50
    updated = 0
    for start in tqdm(range(0, len(images), BATCH), desc="CLIP 批量标注"):
        end = min(start + BATCH, len(images))
        batch_images = images[start:end]
        batch_samples = valid_samples[start:end]

        inputs = processor(
            text=class_texts, images=batch_images,
            return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            probs = outputs.logits_per_image.softmax(dim=1).cpu().numpy()

        for i, sample in enumerate(batch_samples):
            best_idx = int(np.argmax(probs[i]))
            best_class = CLASSES[best_idx]
            best_prob = float(probs[i][best_idx])

            gt_label = label_map.get(best_class, 'other')
            if best_prob < 0.2:
                gt_label = sample.get('seg_prediction', 'other')

            sample['ground_truth'] = gt_label
            sample['clip_label'] = best_class
            sample['clip_confidence'] = round(best_prob, 4)
            updated += 1

    data['label_method'] = 'CLIP zero-shot (openai/clip-vit-base-patch32)'
    with open(TEMPLATE, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"\n标注完成: {updated}/{len(data['samples'])} 个样本")
    print(f"运行评估: python scripts/eval_ground_truth.py")

    # 打印分布
    from collections import Counter
    gt_dist = Counter(s['ground_truth'] for s in data['samples'])
    seg_dist = Counter(s['seg_prediction'] for s in data['samples'])
    print(f"\n{'类别':<12} {'SegFormer':>10} {'CLIP':>10}")
    print("-" * 35)
    for cls in sorted(set(list(gt_dist) + list(seg_dist))):
        print(f"{cls:<12} {seg_dist.get(cls,0):>10} {gt_dist.get(cls,0):>10}")


if __name__ == '__main__':
    main()
