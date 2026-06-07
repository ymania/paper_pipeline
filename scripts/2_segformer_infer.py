#!/usr/bin/env python3
"""
Step 2: SegFormer 语义分割推理 — 论文 §3.3

使用 HuggingFace SegFormer (预训练于 ADE20K) 对每张图像进行语义分割。
输出:
  - intermediate/semantic_maps/   每张图像的分割图 (.npy)
  - intermediate/class_mapping.json  ADE20K 类别名映射
"""
import sys
import json
import os
import logging
import pickle
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ADE20K 150 类名称映射 (完整)
ADE20K_CLASSES = {
    0: 'wall', 1: 'building', 2: 'sky', 3: 'floor', 4: 'tree',
    5: 'ceiling', 6: 'road', 7: 'bed', 8: 'windowpane', 9: 'grass',
    10: 'cabinet', 11: 'sidewalk', 12: 'person', 13: 'earth', 14: 'door',
    15: 'table', 16: 'mountain', 17: 'plant', 18: 'curtain', 19: 'chair',
    20: 'car', 21: 'water', 22: 'painting', 23: 'sofa', 24: 'shelf',
    25: 'house', 26: 'sea', 27: 'mirror', 28: 'rug', 29: 'field',
    30: 'armchair', 31: 'seat', 32: 'fence', 33: 'desk', 34: 'rock',
    35: 'wardrobe', 36: 'lamp', 37: 'bathtub', 38: 'railing', 39: 'cushion',
    40: 'base', 41: 'box', 42: 'column', 43: 'signboard',
    44: 'chest of drawers', 45: 'counter', 46: 'sand', 47: 'sink',
    48: 'skyscraper', 49: 'fireplace', 50: 'refrigerator', 51: 'grandstand',
    52: 'path', 53: 'stairs', 54: 'runway', 55: 'case', 56: 'pool table',
    57: 'pillow', 58: 'screen door', 59: 'stairway', 60: 'river',
    61: 'bridge', 62: 'bookcase', 63: 'blind', 64: 'coffee table',
    65: 'toilet', 66: 'flower', 67: 'book', 68: 'hill', 69: 'bench',
    70: 'countertop', 71: 'stove', 72: 'palm', 73: 'kitchen island',
    74: 'computer', 75: 'swivel chair', 76: 'boat', 77: 'bar',
    78: 'arcade machine', 79: 'hovel', 80: 'bus', 81: 'towel', 82: 'light',
    83: 'truck', 84: 'tower', 85: 'chandelier', 86: 'awning',
    87: 'streetlight', 88: 'booth', 89: 'television', 90: 'airplane',
    91: 'dirt track', 92: 'apparel', 93: 'pole', 94: 'land',
    95: 'bannister', 96: 'escalator', 97: 'ottoman', 98: 'bottle',
    99: 'buffet', 100: 'poster', 101: 'stage', 102: 'van', 103: 'ship',
    104: 'fountain', 105: 'conveyer belt', 106: 'canopy', 107: 'washer',
    108: 'plaything', 109: 'swimming pool', 110: 'stool', 111: 'barrel',
    112: 'basket', 113: 'waterfall', 114: 'tent', 115: 'bag',
    116: 'minibike', 117: 'cradle', 118: 'oven', 119: 'ball', 120: 'food',
    121: 'step', 122: 'tank', 123: 'trade name', 124: 'microwave',
    125: 'pot', 126: 'animal', 127: 'bicycle', 128: 'lake',
    129: 'dishwasher', 130: 'screen', 131: 'blanket', 132: 'sculpture',
    133: 'hood', 134: 'sconce', 135: 'vase', 136: 'traffic light',
    137: 'tray', 138: 'ashcan', 139: 'fan', 140: 'pier',
    141: 'crt screen', 142: 'plate', 143: 'monitor', 144: 'bulletin board',
    145: 'shower', 146: 'radiator', 147: 'glass', 148: 'clock', 149: 'flag',
}


def main():
    config_path = Path(__file__).parent.parent / 'config' / 'paths.json'
    with open(config_path, 'r') as f:
        config = json.load(f)

    data_dir = Path(config['data_root'])
    images_dir = data_dir / config['images_dir']
    output_dir = Path(config['output_root'])
    intermediate = output_dir / 'intermediate'
    semantic_dir = intermediate / 'semantic_maps'
    semantic_dir.mkdir(parents=True, exist_ok=True)

    # 加载 images.pkl 获取图像列表
    with open(intermediate / 'images.pkl', 'rb') as f:
        images = pickle.load(f)

    # 收集所有唯一的图像文件
    image_files = {}
    for img_id, img_info in images.items():
        name = img_info['name']
        img_path = images_dir / name
        if img_path.exists():
            image_files[img_id] = img_path

    logger.info(f"找到 {len(image_files)} 张图像待分割")

    # 加载 SegFormer
    import torch
    from transformers import AutoModelForSemanticSegmentation

    MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"加载模型: {MODEL_NAME} (device={device})")
    model = AutoModelForSemanticSegmentation.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 逐张分割
    for img_id, img_path in tqdm(sorted(image_files.items()), desc="SegFormer 推理"):
        image = Image.open(img_path).convert('RGB')
        orig_w, orig_h = image.size
        
        # 手动预处理绕过 torch.from_numpy bug (torch 2.8 + numpy 1.25)
        image_resized = image.resize((512, 512), Image.BILINEAR)
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        img_array = img_array.transpose(2, 0, 1)  # HWC -> CHW
        pixel_values = torch.as_tensor(img_array, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(pixel_values=pixel_values)
            logits = outputs.logits  # (1, num_classes, 128, 128)

        # 上采样到原图尺寸
        seg_map = torch.nn.functional.interpolate(
            logits,
            size=(orig_h, orig_w),
            mode='bilinear',
            align_corners=False
        )
        seg_map = seg_map[0].argmax(dim=0).cpu().numpy().astype(np.uint8)
        # 强制拷贝确保连续内存布局，避免 numpy object array 陷阱
        seg_map = np.ascontiguousarray(seg_map)

        np.save(semantic_dir / f'{img_id}.npy', seg_map)

    # 保存类别映射
    with open(intermediate / 'class_mapping.json', 'w') as f:
        json.dump(ADE20K_CLASSES, f, indent=2, ensure_ascii=False)

    logger.info(f"语义分割完成: {len(image_files)} 张图像 → {semantic_dir}/")

    return 0


if __name__ == '__main__':
    sys.exit(main())
