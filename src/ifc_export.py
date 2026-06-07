"""
语义实例 → IFC 标准输出 — 论文 §3.6

IFC4 层级结构 (§3.6.3):
  IfcProject → IfcSite → IfcBuilding → IfcBuildingStorey → 建筑构件

ADE20K → IFC 映射规则 (§3.6.2):
  wall, wall-panel, wall-brick  → IfcWall
  floor, floor-marble           → IfcSlab
  door                          → IfcDoor
  window, windowpane            → IfcWindow
  ceiling                       → IfcCovering
  roof                          → IfcRoof
  column                        → IfcColumn
"""
import numpy as np
import logging
import json
from typing import Dict, List, Optional
from pathlib import Path

from .clustering import OBB
from .coord_transform import transform_to_ifc_space

logger = logging.getLogger(__name__)

# 默认 ADE20K → IFC 实体映射
DEFAULT_MAPPING = {
    'wall': 'IfcWall',
    'building': 'IfcBuildingElementProxy',
    'windowpane': 'IfcWindow',
    'window': 'IfcWindow',
    'door': 'IfcDoor',
    'floor': 'IfcSlab',
    'ceiling': 'IfcCovering',
    'roof': 'IfcRoof',
    'column': 'IfcColumn',
    'stairs': 'IfcStair',
    'stairway': 'IfcStair',
}


def load_label_mapping(config_path: str) -> Dict[str, str]:
    """从配置文件加载语义→IFC映射。"""
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            mapping = json.load(f)
        logger.info(f"加载映射规则: {len(mapping)} 条")
        return mapping
    return DEFAULT_MAPPING


def _make_point(ifc_file, coords):
    """创建 IfcCartesianPoint (accepts list/tuple of floats)."""
    return ifc_file.create_entity('IfcCartesianPoint',
        Coordinates=tuple(float(v) for v in coords))

def _make_dir(ifc_file, ratios):
    return ifc_file.create_entity('IfcDirection',
        DirectionRatios=tuple(float(v) for v in ratios))

def _make_placement3d(ifc_file, origin=(0.0,0.0,0.0)):
    return ifc_file.create_entity('IfcAxis2Placement3D',
        Location=_make_point(ifc_file, origin),
        Axis=_make_dir(ifc_file, (0,0,1)),
        RefDirection=_make_dir(ifc_file, (1,0,0)))


def export_ifc(
    instances: List[OBB],
    points: np.ndarray,
    mapping: Dict[str, str],
    output_path: str,
    project_name: str = "SemanticBIM_Project",
    coord_config: dict = None
) -> str:
    """将实例集合导出为 IFC4 文件。"""
    try:
        import ifcopenshell
    except ImportError:
        logger.error("ifcopenshell 未安装。安装: pip install ifcopenshell")
        raise

    import ifcopenshell.api

    # 1. 创建文件 + 项目
    ifc_file = ifcopenshell.api.run('project.create_file', version='IFC4')
    project = ifcopenshell.api.run('root.create_entity', ifc_file,
        ifc_class='IfcProject', name=project_name)
    ifcopenshell.api.run('unit.assign_unit', ifc_file,
        units=[ifc_file.create_entity('IfcSIUnit',
            UnitType='LENGTHUNIT', Name='METRE')])

    # 2. 空间层级
    site = ifcopenshell.api.run('root.create_entity', ifc_file,
        ifc_class='IfcSite', name='DefaultSite')
    building = ifcopenshell.api.run('root.create_entity', ifc_file,
        ifc_class='IfcBuilding', name='DefaultBuilding')
    storey = ifcopenshell.api.run('root.create_entity', ifc_file,
        ifc_class='IfcBuildingStorey', name='GroundFloor')
    ifcopenshell.api.run('aggregate.assign_object', ifc_file,
        relating_object=project, products=[site])
    ifcopenshell.api.run('aggregate.assign_object', ifc_file,
        relating_object=site, products=[building])
    ifcopenshell.api.run('aggregate.assign_object', ifc_file,
        relating_object=building, products=[storey])

    # 3. OwnerHistory
    org = ifc_file.create_entity('IfcOrganization', Name='Hermes Pipeline')
    person = ifc_file.create_entity('IfcPerson', GivenName='Researcher')
    person_and_org = ifc_file.create_entity('IfcPersonAndOrganization',
        ThePerson=person, TheOrganization=org)
    application = ifc_file.create_entity('IfcApplication',
        ApplicationDeveloper=org, Version='1.0',
        ApplicationFullName='Semantic BIM Pipeline',
        ApplicationIdentifier='paper_pipeline')
    owner_history = ifc_file.create_entity('IfcOwnerHistory',
        OwningUser=person_and_org, OwningApplication=application,
        ChangeAction='ADDED', CreationDate=0)

    # 4. 几何上下文
    geom_ctx = ifc_file.create_entity('IfcGeometricRepresentationContext',
        ContextIdentifier='Body', ContextType='Model',
        CoordinateSpaceDimension=3, Precision=0.001,
        WorldCoordinateSystem=_make_placement3d(ifc_file))

    # 5. 逐实例创建
    created = 0
    for inst in instances:
        ifc_type = mapping.get(inst.class_name)
        if ifc_type is None:
            for key, val in mapping.items():
                if key in inst.class_name:
                    ifc_type = val
                    break
            if ifc_type is None:
                ifc_type = 'IfcBuildingElementProxy'

        try:
            center = [float(v) for v in inst.center]
            dims = inst.dimensions
            half_w = float(dims[0]) / 2.0
            half_d = float(dims[1]) / 2.0
            height = float(dims[2]) if float(dims[2]) > 0.01 else 0.1

            placement = ifc_file.create_entity('IfcLocalPlacement',
                RelativePlacement=_make_placement3d(ifc_file, center))

            profile = ifc_file.create_entity('IfcRectangleProfileDef',
                ProfileType='AREA',
                ProfileName=f'{inst.class_name}_profile',
                Position=ifc_file.create_entity('IfcAxis2Placement2D',
                    Location=_make_point(ifc_file, (0,0)),
                    RefDirection=_make_dir(ifc_file, (1,0))),
                XDim=2.0 * half_w,
                YDim=2.0 * half_d)

            body = ifc_file.create_entity('IfcExtrudedAreaSolid',
                SweptArea=profile,
                Position=_make_placement3d(ifc_file),
                ExtrudedDirection=_make_dir(ifc_file, (0,0,1)),
                Depth=height)

            shape_repr = ifc_file.create_entity('IfcShapeRepresentation',
                ContextOfItems=geom_ctx,
                RepresentationIdentifier='Body',
                RepresentationType='SweptSolid',
                Items=[body])

            product_def = ifc_file.create_entity('IfcProductDefinitionShape',
                Representations=[shape_repr])

            entity = ifc_file.create_entity(ifc_type,
                GlobalId=ifcopenshell.guid.new(),
                OwnerHistory=owner_history,
                Name=f'{inst.class_name}_{inst.instance_id}',
                ObjectPlacement=placement,
                Representation=product_def)

            ifc_file.create_entity('IfcRelContainedInSpatialStructure',
                GlobalId=ifcopenshell.guid.new(),
                RelatedElements=[entity],
                RelatingStructure=storey)

            created += 1

        except Exception as e:
            logger.warning(f"实例 {inst.class_name}_{inst.instance_id} 创建失败: {e}")
            continue

    # 6. 写入
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ifc_file.write(str(out))

    logger.info(f"IFC 导出完成: {created}/{len(instances)} 个实体 → {out}")
    logger.info(f"文件大小: {out.stat().st_size / 1024:.1f} KB")

    return str(out)
