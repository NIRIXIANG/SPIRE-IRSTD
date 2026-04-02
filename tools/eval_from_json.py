#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
红外小目标检测指标评估工具（独立版本，仅依赖 Python 标准库）

使用方法:
    python eval_from_json.py --gt gt.json --pred pred.json
    python eval_from_json.py --gt gt.json --pred pred.json --output results.txt
    python eval_from_json.py --gt gt.json --pred pred.json --tp-distance 3.0
    python eval_from_json.py --gt gt.json --pred pred.json --thresholds 0.1 0.3 0.5 0.7 0.9

参数说明:
    --gt: GT 关键点 JSON 文件路径（COCO 格式）
    --pred: 预测关键点 JSON 文件路径（COCO 格式）
    --tp-distance: TP 距离阈值，默认 5.0
    --thresholds: 阈值列表，默认 [0.1, 0.2, ..., 1.0]
    --output: 结果输出文件路径（可选）

【重要】图像尺寸自动从 JSON 的 images 字段读取，用于 FA 计算
"""

import argparse
import json
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

# ============================================================
# 核心算法：GT 驱动的关键点匹配
# ============================================================

def euclidean_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    """计算两点间的欧氏距离"""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def match_keypoints_gt_driven(
    pred_keypoints: List[Tuple[float, float, float]],
    gt_keypoints: List[Tuple[float, float]],
    distance_threshold: float
) -> Tuple[int, int, int, set]:
    """
    GT 驱动的一对一匹配算法
    
    匹配策略:
    - 以 GT 为中心做匹配
    - 每个预测点最多匹配一个 GT
    - 每个 GT 寻找最近的未匹配预测点
    
    Returns:
        (tp, fp, fn, matched_pred_indices)
    """
    # 过滤无效的 [0, 0] 预测点
    valid_pred = [(x, y, s) for x, y, s in pred_keypoints if not (x == 0 and y == 0)]
    
    gt_count = len(gt_keypoints)
    pred_count = len(valid_pred)
    
    # 边界情况
    if gt_count == 0:
        return 0, pred_count, 0, set()
    if pred_count == 0:
        return 0, 0, gt_count, set()
    
    # GT 驱动匹配
    tp = 0
    pred_matched = [False] * pred_count
    matched_indices = set()
    
    for gt_x, gt_y in gt_keypoints:
        best_idx = -1
        best_dist = float('inf')
        
        for i, (px, py, _) in enumerate(valid_pred):
            if pred_matched[i]:
                continue
            dist = euclidean_distance((gt_x, gt_y), (px, py))
            if dist < best_dist:
                best_dist = dist
                best_idx = i
        
        if best_idx >= 0 and best_dist <= distance_threshold:
            tp += 1
            pred_matched[best_idx] = True
            matched_indices.add(best_idx)
    
    fp = sum(1 for m in pred_matched if not m)
    fn = gt_count - tp
    
    return tp, fp, fn, matched_indices


# ============================================================
# JSON 加载与解析
# ============================================================

def load_coco_keypoints(json_path: str) -> Tuple[Dict, Dict]:
    """
    加载 COCO 格式的关键点 JSON
    
    Returns:
        (keypoints_dict, image_sizes_dict)
        - keypoints_dict: {image_id: [(x, y, score), ...]}
        - image_sizes_dict: {image_id: (width, height)}
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    def norm_id(img_id):
        """统一 ID 格式，去掉扩展名"""
        return os.path.splitext(str(img_id))[0]
    
    # 解析图像尺寸
    image_sizes = {}
    for img in data.get('images', []):
        img_id = norm_id(img.get('id', ''))
        w, h = img.get('width', 0), img.get('height', 0)
        if w > 0 and h > 0:
            image_sizes[img_id] = (w, h)
    
    # 解析关键点
    keypoints = {}
    for ann in data.get('annotations', []):
        img_id = norm_id(ann.get('image_id', ''))
        kpts = ann.get('keypoints', [])
        
        # 解析 [x, y, score, x, y, score, ...] 格式
        points = []
        for i in range(0, len(kpts), 3):
            if i + 2 < len(kpts):
                points.append((float(kpts[i]), float(kpts[i+1]), float(kpts[i+2])))
        
        keypoints[img_id] = points
    
    return keypoints, image_sizes


# ============================================================
# 评估核心逻辑
# ============================================================

def evaluate_from_json(
    gt_json_path: str,
    pred_json_path: str,
    tp_distance: float = 5.0,
    pdfa_thresholds: Optional[List[float]] = None
) -> Dict:
    """
    从 JSON 文件计算评价指标
    
    Args:
        gt_json_path: GT JSON 文件路径
        pred_json_path: 预测 JSON 文件路径
        tp_distance: TP 距离阈值
        pdfa_thresholds: 置信度阈值列表
    
    Returns:
        包含 metrics, pdfa_table, bad_image_ids 的字典
    """
    # 默认阈值
    if pdfa_thresholds is None:
        pdfa_thresholds = [round(i * 0.1, 1) for i in range(1, 11)]
    
    # 加载数据
    gt_data, gt_sizes = load_coco_keypoints(gt_json_path)
    pred_data, pred_sizes = load_coco_keypoints(pred_json_path)
    
    # 合并图像尺寸（GT 优先）
    all_sizes = {**pred_sizes, **gt_sizes}
    
    # 以 pred json 中的图像为准进行评估
    all_ids = sorted(pred_data.keys())
    
    # 存储每张图像的匹配结果
    per_image_results = []
    match_cache = []  # 用于多阈值计算
    total_pixels = 0.0
    
    # 基础指标累加（阈值=0）
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for img_id in all_ids:
        gt_kpts = gt_data.get(img_id, [])
        pred_kpts = pred_data.get(img_id, [])
        
        # 提取 GT 坐标 (x, y)
        gt_coords = [(x, y) for x, y, _ in gt_kpts] if gt_kpts else []
        
        # 匹配
        tp, fp, fn, matched_indices = match_keypoints_gt_driven(
            pred_kpts, gt_coords, tp_distance
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 记录结果
        per_image_results.append({
            'image_id': img_id,
            'tp': tp, 'fp': fp, 'fn': fn,
            'num_gt': len(gt_coords),
            'num_pred': len(pred_kpts)
        })
        
        # 缓存匹配状态（用于多阈值计算）
        # 过滤无效预测点后的置信度列表
        valid_pred = [(x, y, s) for x, y, s in pred_kpts if not (x == 0 and y == 0)]
        scores = [s for _, _, s in valid_pred]
        match_cache.append({
            'matched_indices': matched_indices,
            'scores': scores,
            'num_gt': len(gt_coords),
            'num_pred': len(valid_pred)
        })
        
        # 累加像素数
        if img_id in all_sizes:
            w, h = all_sizes[img_id]
            total_pixels += w * h
        else:
            total_pixels += 512 * 640  # 默认尺寸
    
    # 计算基础指标
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    metrics = {
        'tp': total_tp,
        'fp': total_fp,
        'fn': total_fn,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'image_count': len(all_ids)
    }
    
    # 计算多阈值指标
    pdfa_table = compute_multi_threshold_metrics(
        match_cache, pdfa_thresholds, total_pixels
    )
    
    # 获取错误图像列表
    bad_ids = [r['image_id'] for r in per_image_results if r['fp'] > 0 or r['fn'] > 0]
    
    return {
        'metrics': metrics,
        'pdfa_table': pdfa_table,
        'per_image': per_image_results,
        'bad_image_ids': sorted(bad_ids)
    }


def compute_multi_threshold_metrics(
    match_cache: List[Dict],
    thresholds: List[float],
    total_pixels: float
) -> Dict:
    """计算多阈值下的指标"""
    results = {
        'thresholds': thresholds,
        'TP': [], 'FP': [], 'FN': [],
        'Precision': [], 'Recall': [], 'F1': [],
        'PD': [], 'FA': []
    }
    
    for thresh in thresholds:
        tp_sum, fp_sum, fn_sum = 0, 0, 0
        
        for cache in match_cache:
            matched = cache['matched_indices']
            scores = cache['scores']
            num_gt = cache['num_gt']
            
            # 筛选置信度 >= 阈值的预测点
            above_thresh = {i for i, s in enumerate(scores) if s >= thresh}
            
            # TP = 匹配成功 且 置信度 >= 阈值
            tp = len(matched & above_thresh)
            # FP = 置信度 >= 阈值 但 未匹配
            fp = len(above_thresh - matched)
            # FN = GT 总数 - TP
            fn = num_gt - tp
            
            tp_sum += tp
            fp_sum += fp
            fn_sum += fn
        
        # 计算指标
        prec = tp_sum / (tp_sum + fp_sum) if (tp_sum + fp_sum) > 0 else 0.0
        rec = tp_sum / (tp_sum + fn_sum) if (tp_sum + fn_sum) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        fa = fp_sum / total_pixels if total_pixels > 0 else 0.0
        
        results['TP'].append(tp_sum)
        results['FP'].append(fp_sum)
        results['FN'].append(fn_sum)
        results['Precision'].append(prec)
        results['Recall'].append(rec)
        results['F1'].append(f1)
        results['PD'].append(rec)  # PD = Recall
        results['FA'].append(fa)
    
    return results


# ============================================================
# 输出与保存
# ============================================================

def print_pdfa_table(pdfa_table: Dict):
    """打印 PD_FA 阈值表"""
    print("-" * 120)
    print(f"{'Thresh':<8} {'TP':>8} {'FP':>8} {'FN':>8} "
          f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'PD':>10} {'FA':>14}")
    print("-" * 120)
    
    for i, t in enumerate(pdfa_table['thresholds']):
        print(f"{t:<8.2f} {pdfa_table['TP'][i]:>8} {pdfa_table['FP'][i]:>8} "
              f"{pdfa_table['FN'][i]:>8} {pdfa_table['Precision'][i]:>10.6f} "
              f"{pdfa_table['Recall'][i]:>10.6f} {pdfa_table['F1'][i]:>10.6f} "
              f"{pdfa_table['PD'][i]:>10.6f} {pdfa_table['FA'][i]:>14.2e}")
    
    print("-" * 120)


def save_results(results: Dict, output_path: str, tp_distance: float, thresholds: List[float]):
    """保存结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("IRST Metrics Evaluation Results\n")
        f.write("=" * 120 + "\n\n")
        
        # 配置
        f.write("【Configuration】\n")
        f.write("-" * 80 + "\n")
        f.write(f"TP Distance Threshold: {tp_distance}\n")
        f.write(f"Thresholds: {thresholds}\n\n")
        
        # 基本指标
        m = results['metrics']
        f.write("【Overall Metrics】\n")
        f.write("-" * 80 + "\n")
        f.write(f"TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}\n")
        f.write(f"Precision: {m['precision']:.6f}\n")
        f.write(f"Recall: {m['recall']:.6f}\n")
        f.write(f"F1 Score: {m['f1_score']:.6f}\n")
        f.write(f"Image Count: {m['image_count']}\n\n")
        
        # PD_FA 表
        table = results['pdfa_table']
        f.write("【PD_FA Threshold Table】\n")
        f.write("-" * 120 + "\n")
        f.write(f"{'Thresh':<10} {'TP':>8} {'FP':>8} {'FN':>8} "
                f"{'Precision':>12} {'Recall':>12} {'F1':>12} {'FA':>15}\n")
        f.write("-" * 120 + "\n")
        
        for i, t in enumerate(table['thresholds']):
            f.write(f"{t:<10.2f} {table['TP'][i]:>8} {table['FP'][i]:>8} "
                    f"{table['FN'][i]:>8} {table['Precision'][i]:>12.6f} "
                    f"{table['Recall'][i]:>12.6f} {table['F1'][i]:>12.6f} "
                    f"{table['FA'][i]:>15.10f}\n")
        f.write("\n")
        
        # 错误图像
        bad_ids = results['bad_image_ids']
        f.write(f"【Error Images】({len(bad_ids)} total)\n")
        f.write("-" * 80 + "\n")
        if bad_ids:
            f.write(", ".join(bad_ids[:100]))
            if len(bad_ids) > 100:
                f.write(f" ... (and {len(bad_ids) - 100} more)")
            f.write("\n")


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='红外小目标检测指标评估工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--gt', required=True, help='GT JSON 文件路径')
    parser.add_argument('--pred', required=True, help='预测 JSON 文件路径')
    parser.add_argument('--tp-distance', type=float, default=5.0, help='TP 距离阈值 (默认: 5.0)')
    parser.add_argument('--thresholds', type=float, nargs='*', default=None,
                        help='阈值列表 (默认: 0.1-1.0 间隔 0.1)')
    parser.add_argument('--output', type=str, default=None, help='结果输出文件路径')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查文件
    if not os.path.exists(args.gt):
        print(f"Error: GT file not found: {args.gt}")
        sys.exit(1)
    if not os.path.exists(args.pred):
        print(f"Error: Prediction file not found: {args.pred}")
        sys.exit(1)
    
    # 设置阈值
    thresholds = args.thresholds or [round(i * 0.1, 1) for i in range(1, 11)]
    
    # 打印头部
    print("=" * 120)
    print("IRST Metrics Evaluation from JSON")
    print("=" * 120)
    print(f"GT file: {args.gt}")
    print(f"Pred file: {args.pred}")
    print(f"TP distance: {args.tp_distance}")
    print(f"Thresholds: {thresholds}")
    print("-" * 120)
    
    # 评估
    results = evaluate_from_json(
        args.gt, args.pred,
        tp_distance=args.tp_distance,
        pdfa_thresholds=thresholds
    )
    
    # 打印基本指标
    m = results['metrics']
    print(f"Overall Metrics (threshold=0):")
    print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")
    print(f"  Precision: {m['precision']:.6f}")
    print(f"  Recall: {m['recall']:.6f}")
    print(f"  F1 Score: {m['f1_score']:.6f}")
    print(f"  Image Count: {m['image_count']}")
    
    # 打印阈值表
    print("\nPD_FA Threshold Table:")
    print_pdfa_table(results['pdfa_table'])
    
    # 找出最佳 F1
    f1_list = results['pdfa_table']['F1']
    best_idx = f1_list.index(max(f1_list))
    best_thresh = results['pdfa_table']['thresholds'][best_idx]
    print(f"\n★ Best F1: {f1_list[best_idx]:.6f} @ threshold={best_thresh:.2f}")
    
    # 打印错误图像
    bad_ids = results['bad_image_ids']
    if bad_ids:
        print(f"\nError images ({len(bad_ids)} total): {', '.join(bad_ids[:20])}", end='')
        if len(bad_ids) > 20:
            print(f" ... (and {len(bad_ids) - 20} more)")
        else:
            print()
    
    print("=" * 120)
    
    # 保存结果
    if args.output:
        save_results(results, args.output, args.tp_distance, thresholds)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
