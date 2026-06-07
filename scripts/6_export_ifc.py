#!/usr/bin/env python3
"""
Step 6: IFC 模型导出 — 论文 §3.6

将 DBSCAN 提取的构件实例转换为 IFC4 标准文件。
使用 IfcOpenShell 生成符合 ISO 16739 标准的 BIM 模型。

IFC 层级:
  IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey → 构件

输出:
  - output/semantic_bim.ifc
"""
import sys
import json
import logging
import pickle
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.ifc_export import export_ifc, load_label_mapping

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    project_root = Path(__file__).resolve().parent.parent

    with open(project_root / 'config' / 'pipeline_params.json', 'r') as f:
        params = json.load(f)

    output_dir = Path('output') if not (project_root / 'config' / 'paths.json').exists() else \
        Path(json.load(open(project_root / 'config' / 'paths.json'))['output_root'])
    intermediate = output_dir / 'intermediate'

    # 加载实例
    with open(intermediate / 'instances.pkl', 'rb') as f:
        instances = pickle.load(f)

    if not instances:
        logger.error("没有实例可导出。请先运行 step 5。")
        return 1

    # 加载映射规则
    mapping_path = project_root / 'config' / 'label_to_ifc.json'
    mapping = load_label_mapping(str(mapping_path))

    # 加载点云 (供坐标变换)
    import numpy as np
    points = np.load(intermediate / 'sparse_points.npy')

    # 导出 IFC
    ifc_path = output_dir / 'semantic_bim.ifc'
    export_ifc(
        instances=instances,
        points=points,
        mapping=mapping,
        output_path=str(ifc_path),
        project_name='SemanticBIM_from_COLMAP'
    )

    logger.info(f"IFC 文件: {ifc_path}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
