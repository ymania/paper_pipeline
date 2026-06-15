#!/usr/bin/env python3
"""
消融实验 — §4.4 改进
自动跑 6 组配置，对比加权投票策略效果

配置:
  A: α=1.0 (纯距离权重 — 当前 baseline)
  B: α=0.5 (距离+角度联合 — 本文方法)
  C: α=0.0 (纯角度权重)
  D: α=0.5 + LOF=off + 对齐=off (只加权投票)
  E: α=0.5 + LOF=on + 对齐=off
  F: α=0.5 + LOF=on + 对齐=on (完整方法)

输出: output/ablation_results.json + ablation_table.md
"""
import sys, json, logging, pickle, subprocess, time
import numpy as np
from pathlib import Path
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

PROJECT = Path(__file__).resolve().parent.parent
SCRIPTS = PROJECT / 'scripts'
INTERMEDIATE = PROJECT / 'output' / 'intermediate'
CONFIG_PATH = PROJECT / 'config' / 'pipeline_params.json'
PYTHON = '/home/qmyyc/miniconda3/envs/ms/bin/python'


def update_config(alpha=None, use_lof=None, use_alignment=None, eps=None):
    """修改 pipeline_params.json 中的参数"""
    with open(CONFIG_PATH, 'r') as f:
        cfg = json.load(f)

    if alpha is not None:
        cfg['voting']['distance_alpha'] = alpha

    if use_lof is not None:
        cfg['clustering']['use_lof'] = use_lof

    if use_alignment is not None:
        cfg['clustering']['use_alignment'] = use_alignment

    if eps is not None:
        cfg['clustering']['eps'] = eps

    with open(CONFIG_PATH, 'w') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    return cfg


def run_step4(alpha_label):
    """跑加权投票 (step 4)"""
    t0 = time.time()
    result = subprocess.run(
        [PYTHON, str(SCRIPTS / '4_weighted_voting.py')],
        capture_output=True, text=True, cwd=PROJECT, timeout=120
    )
    duration = time.time() - t0

    if result.returncode != 0:
        logger.error(f"Step 4 ({alpha_label}) 失败: {result.stderr[-500:]}")
        return None

    # 解析输出 (logging → stderr)
    stats = {}
    output = result.stdout + '\n' + result.stderr
    for line in output.split('\n'):
        if '加权投票结果' in line:
            parts = line.split('/')
            if len(parts) >= 2:
                stats['n_labeled'] = int(parts[0].split(':')[1].strip())
        if '唯一类别数' in line:
            stats['n_classes'] = int(line.split(':')[1].strip())
        if '平均置信度' in line:
            stats['mean_conf'] = float(line.split(':')[1].strip())
        if '标签一致性' in line:
            # Format: "标签一致性 (置信度>0.5): 100.0%"
            try:
                stats['consistency'] = float(line.strip().split(':')[-1].replace('%', '').strip())
            except ValueError:
                pass

    stats['duration_s'] = round(duration, 1)
    return stats


def run_step5():
    """跑实例分割 (step 5)"""
    result = subprocess.run(
        [PYTHON, str(SCRIPTS / '5_instance_cluster.py')],
        capture_output=True, text=True, cwd=PROJECT, timeout=60
    )
    if result.returncode != 0:
        logger.error(f"Step 5 失败: {result.stderr[-500:]}")
        return {}

    stats = {}
    output = result.stdout + '\n' + result.stderr
    lines = output.split('\n')
    for line in lines:
        if 'LOF 总计剔除' in line:
            stats['lof_removed'] = int(line.split(':')[1].strip().split()[0])
        if '实例分割完成' in line and '个实例' in line:
            # Format: "实例分割完成: 17 个实例, 2125 个点分配 (95.0%)"
            parts = line.split(',')
            try:
                stats['n_instances'] = int(parts[0].split(':')[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
            if len(parts) >= 2:
                try:
                    stats['assigned_ratio'] = float(parts[1].split('(')[1].replace('%', '').replace(')', '').strip())
                except (ValueError, IndexError):
                    pass
        if '主方向对齐完成' in line:
            stats['pca_aligned'] = True

    return stats


def collect_results(run_name, alpha_stats, cluster_stats):
    """汇总一次实验的结果"""
    return {
        'name': run_name,
        'alpha': cluster_stats.get('_alpha', None),
        'use_lof': cluster_stats.get('_use_lof', None),
        'use_alignment': cluster_stats.get('_use_alignment', None),
        'n_labeled': alpha_stats.get('n_labeled', 0),
        'n_classes': alpha_stats.get('n_classes', 0),
        'mean_confidence': alpha_stats.get('mean_conf', 0),
        'consistency_pct': alpha_stats.get('consistency', 0),
        'voting_duration_s': alpha_stats.get('duration_s', 0),
        'lof_removed': cluster_stats.get('lof_removed', 0),
        'n_instances': cluster_stats.get('n_instances', 0),
        'instance_assigned_ratio': cluster_stats.get('assigned_ratio', 0),
    }


def main():
    # 备份原始配置
    with open(CONFIG_PATH, 'r') as f:
        orig_cfg = json.load(f)
    orig_cfg_str = json.dumps(orig_cfg, indent=2)

    experiments = [
        # (name, alpha, use_lof, use_alignment)
        ('A-纯距离', 1.0, False, False),
        ('B-联合权重', 0.5, False, False),
        ('C-纯角度', 0.0, False, False),
        ('D-联合+LOF', 0.5, True, False),
        ('E-联合+对齐', 0.5, False, True),
        ('F-完整方法', 0.5, True, True),
    ]

    results = []
    logger.info(f"开始消融实验 — {len(experiments)} 组配置")
    logger.info("=" * 70)

    for name, alpha, use_lof, use_alignment in experiments:
        logger.info(f"\n{'='*50}")
        logger.info(f"  {name}: α={alpha}, LOF={'✓' if use_lof else '✗'}, 对齐={'✓' if use_alignment else '✗'}")
        logger.info(f"{'='*50}")

        # 更新配置
        cfg = update_config(alpha=alpha, use_lof=use_lof, use_alignment=use_alignment)

        # Step 4: 加权投票
        alpha_stats = run_step4(name)
        if alpha_stats is None:
            logger.warning(f"  {name} step4 失败，跳过")
            continue
        logger.info(f"  投票: {alpha_stats['n_labeled']} 点, {alpha_stats['n_classes']} 类, "
                    f"一致性={alpha_stats['consistency']}%")

        # Step 5: 实例分割
        cluster_stats = run_step5()
        cluster_stats['_alpha'] = alpha
        cluster_stats['_use_lof'] = use_lof
        cluster_stats['_use_alignment'] = use_alignment

        logger.info(f"  聚类: {cluster_stats.get('n_instances', '?')} 实例, "
                    f"LOF剔除={cluster_stats.get('lof_removed', 0)}, "
                    f"分配率={cluster_stats.get('assigned_ratio', 0)}%")

        result = collect_results(name, alpha_stats, cluster_stats)
        results.append(result)

    # 恢复原始配置
    with open(CONFIG_PATH, 'w') as f:
        f.write(orig_cfg_str)

    # 保存结果
    output = {
        'timestamp': datetime.now().isoformat(),
        'experiments': results,
    }

    out_dir = PROJECT / 'output'
    with open(out_dir / 'ablation_results.json', 'w') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    # 生成 Markdown 表格
    md = [
        "# 消融实验结果",
        "",
        f"实验时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "| 配置 | α | LOF | 对齐 | 语义点 | 类别数 | 一致性(%) | 平均置信度 | 实例数 | LOF剔除 | 分配率(%) |",
        "|------|---|-----|------|--------|--------|-----------|-----------|--------|---------|----------|",
    ]

    for r in results:
        lof = '✓' if r['use_lof'] else '✗'
        align = '✓' if r['use_alignment'] else '✗'
        md.append(
            f"| {r['name']} | {r['alpha']} | {lof} | {align} | "
            f"{r['n_labeled']} | {r['n_classes']} | "
            f"{r['consistency_pct']}% | {r['mean_confidence']:.4f} | "
            f"{r['n_instances']} | {r['lof_removed']} | "
            f"{r['instance_assigned_ratio']:.1f}% |"
        )

    with open(out_dir / 'ablation_table.md', 'w') as f:
        f.write('\n'.join(md))

    logger.info(f"\n消融实验结果 → {out_dir / 'ablation_table.md'}")
    logger.info("\n" + '\n'.join(md[-len(results):]))

    return 0


if __name__ == '__main__':
    sys.exit(main())
