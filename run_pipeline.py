#!/usr/bin/env python3
"""
论文复现管道总调度 — 7 步端到端执行

  1. COLMAP 数据解析         (§3.2)
  2. SegFormer 语义分割      (§3.3)
  3. 3D→2D 投影可见性        (§3.4.1-2)
  4. 几何引导加权投票         (§3.4.3-6) ★ 核心创新
  5. DBSCAN 实例分割 + OBB   (§3.5)     ★ 核心创新
  6. IFC 模型导出             (§3.6)     ★ 核心创新
  7. 可视化 + 统计            (附录)

用法:
  python run_pipeline.py              # 完整运行
  python run_pipeline.py --step 4     # 只运行第4步
  python run_pipeline.py --from 3     # 从第3步开始
"""
import sys
import subprocess
import logging
import argparse
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent / 'scripts'

STEPS = [
    ("1_colmap_parse.py",      "COLMAP 数据解析",           "§3.2"),
    ("2_segformer_infer.py",   "SegFormer 语义分割",        "§3.3"),
    ("3_projection.py",        "3D→2D 投影与可见性",         "§3.4.1-2"),
    ("4_weighted_voting.py",   "几何引导加权投票 ★",         "§3.4.3-6"),
    ("5_instance_cluster.py",  "DBSCAN 实例分割 + OBB ★",   "§3.5"),
    ("6_export_ifc.py",        "IFC4 模型导出 ★",           "§3.6"),
    ("7_visualize.py",         "可视化与统计",               "附录"),
]


def get_python():
    """获取正确的 Python 解释器 (conda env ms 或其他)"""
    import shutil
    # 优先使用 conda env ms
    conda_python = '/home/qmyyc/miniconda3/envs/ms/bin/python3'
    if Path(conda_python).exists():
        return conda_python
    # 回退到当前 Python
    return sys.executable


def run_script(script_name: str, description: str, citation: str) -> bool:
    script_path = SCRIPT_DIR / script_name
    if not script_path.exists():
        logger.error(f"脚本缺失: {script_path}")
        return False

    logger.info(f"\n{'='*60}")
    logger.info(f"  {description}  [{citation}]")
    logger.info(f"{'='*60}")

    result = subprocess.run(
        [get_python(), str(script_path)],
        capture_output=False,
        cwd=Path(__file__).resolve().parent
    )

    if result.returncode != 0:
        logger.error(f"❌ 步骤失败 (exit={result.returncode})")
        return False

    logger.info(f"✅ {description} 完成\n")
    return True


def main():
    parser = argparse.ArgumentParser(description='论文复现管道')
    parser.add_argument('--step', type=int, choices=range(1, 8), help='只运行指定步骤')
    parser.add_argument('--from', dest='from_step', type=int, choices=range(1, 8), help='从指定步骤开始')
    parser.add_argument('--list', action='store_true', help='列出所有步骤')
    args = parser.parse_args()

    if args.list:
        for i, (script, desc, cite) in enumerate(STEPS, 1):
            print(f"  {i}. {desc:30s} {cite:15s} → {script}")
        return 0

    logger.info("╔══════════════════════════════════════════╗")
    logger.info("║  基于多视图融合的稀疏语义点云生成        ║")
    logger.info("║  与BIM轻量化表达 — 论文复现管道          ║")
    logger.info("╚══════════════════════════════════════════╝")

    if args.step:
        scripts = [STEPS[args.step - 1]]
    elif args.from_step:
        scripts = STEPS[args.from_step - 1:]
    else:
        scripts = STEPS

    for script, desc, cite in scripts:
        success = run_script(script, desc, cite)
        if not success:
            logger.error("管道中断。")
            return 1

    logger.info("\n╔══════════════════════════════════════════╗")
    logger.info("║  全流程完成 ✅                           ║")
    logger.info("║  输出: output/semantic_bim.ifc           ║")
    logger.info("║        output/stats.json                 ║")
    logger.info("║        output/visualization.html         ║")
    logger.info("╚══════════════════════════════════════════╝")
    return 0


if __name__ == '__main__':
    sys.exit(main())
