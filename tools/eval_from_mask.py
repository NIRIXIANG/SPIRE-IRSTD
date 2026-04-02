#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
红外小目标检测指标评估工具（Mask/热力图 版本）

从 GT masks 和预测 masks/热力图 中通过八连通聚类提取目标，计算评价指标。
与 eval_from_json.py 使用相同的指标计算逻辑。

【支持的输入格式】
- 二值 mask: 0/1 或 0/255
- 热力图: 0-1 范围（浮点数）或 0-255 范围（整数）
- 自动检测输入范围并统一归一化到 0-1

使用方法:
    python eval_from_mask.py --gt-masks gt_masks/ --pred-masks pred_masks/ --images images/
    python eval_from_mask.py --gt-masks gt_masks/ --pred-masks pred_masks/ --images images/ --output results.txt
    python eval_from_mask.py --gt-masks gt_masks/ --pred-masks pred_masks/ --images images/ --tp-distance 5.0
    python eval_from_mask.py --gt-masks gt_masks/ --pred-masks pred_masks/ --images images/ --thresholds 0.1 0.3 0.5 0.7 0.9
    例如: python tools\eval_from_mask.py --gt-masks N:\AcademicResearchs\Datasets\IRSingleFrame_Dataset\SIRST-UAVB\SIRST-UAVB_OnlyUAV_Form\masks 
    --pred-masks N:\AcademicResearchs\Projects\IRSTD\BasicIRSTD-main\results\SIRST-onlyUAVB\ACM 
    --images N:\AcademicResearchs\Datasets\IRSingleFrame_Dataset\SIRST-UAVB\SIRST-UAVB_OnlyUAV_Form\images
    --thresholds 1 --output N:\AcademicResearchs\Experiments\LiTENetExp\ACM\EvaResults\acm_results.txt

参数说明:
    --gt-masks: GT mask 目录（可包含 train+test 全部图像）
    --pred-masks: 预测 mask/热力图 目录（以此为基准确定评估图像数量）
    --images: 原始图像目录（用于获取图像尺寸，可包含 train+test 全部图像）
    --tp-distance: TP 距离阈值，默认 5.0
    --thresholds: 置信度阈值列表，默认 [0.1, 0.2, ..., 1.0]
    --output: 结果输出文件路径（可选）

【注意】评估图像数量以 pred-masks 目录中的图像为准

【重要】
- 使用 skimage.measure.label 进行八连通聚类
- 目标中心点通过 regionprops 的 centroid 或加权质心计算
- 目标置信度 = 目标区域内的最大像素值（归一化后）
- 支持 0-1 和 0-255 两种热力图范围，自动检测并归一化
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

try:
    from skimage import measure
except ImportError:
    print("Error: scikit-image is required. Install with: pip install scikit-image")
    sys.exit(1)


# ============================================================
# 从 eval_from_json.py 复用的核心算法
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
    GT 驱动的一对一匹配算法（与 eval_from_json.py 完全相同）
    
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


def compute_multi_threshold_metrics(
    match_cache: List[Dict],
    thresholds: List[float],
    total_pixels: float
) -> Dict:
    """计算多阈值下的指标（与 eval_from_json.py 完全相同）"""
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
# Mask 特有的目标提取函数
# ============================================================

def normalize_mask(mask_arr: np.ndarray) -> np.ndarray:
    """
    自动检测 mask 范围并归一化到 0-1
    
    支持的输入范围:
    - 0-1 (浮点数热力图)
    - 0-255 (8位整数热力图)
    - 二值 mask (0/1 或 0/255)
    
    Returns:
        归一化到 0-1 范围的 float32 数组
    """
    mask_float = mask_arr.astype(np.float32)
    
    max_val = mask_float.max()
    
    if max_val == 0:
        # 空 mask
        return mask_float
    elif max_val <= 1.0:
        # 已经是 0-1 范围
        return mask_float
    elif max_val <= 255:
        # 0-255 范围，归一化到 0-1
        return mask_float / 255.0
    else:
        # 其他范围，归一化到 0-1
        return mask_float / max_val


def extract_targets_from_mask(
    mask_path: str, 
    use_weighted_centroid: bool = True,
    binary_threshold: float = 0.0
) -> List[Tuple[float, float, float]]:
    """
    从 mask/热力图中使用八连通聚类提取目标中心点
    
    Args:
        mask_path: mask 文件路径
        use_weighted_centroid: 是否使用加权质心（基于像素强度）
        binary_threshold: 二值化阈值（0-1 范围），用于提取连通区域
    
    Returns:
        目标列表 [(x, y, score), ...]
        - x, y: 目标中心坐标
        - score: 置信度（区域内最大像素值，0-1 范围）
    """
    # 读取 mask
    mask = Image.open(mask_path)
    mask_arr = np.array(mask)
    
    # 如果是多通道，取第一个通道
    if len(mask_arr.shape) == 3:
        mask_arr = mask_arr[:, :, 0]
    
    # 归一化到 0-1 范围
    mask_norm = normalize_mask(mask_arr)
    
    # 二值化（用于聚类）
    mask_bin = (mask_norm > binary_threshold).astype(np.uint8)
    
    # 八连通聚类
    labeled = measure.label(mask_bin, connectivity=2)
    props = measure.regionprops(labeled, intensity_image=mask_norm)
    
    targets = []
    for prop in props:
        # 默认使用几何质心
        centroid = prop.centroid
        
        # 尝试使用加权质心
        if use_weighted_centroid:
            try:
                weighted = prop.centroid_weighted
                if weighted is not None and len(weighted) >= 2:
                    centroid = weighted
            except (AttributeError, TypeError):
                pass  # 使用几何质心
        
        # 提取坐标（注意：regionprops 返回 (row, col) 即 (y, x)）
        cy, cx = centroid[0], centroid[1]
        
        # 置信度 = 区域内最大像素值（归一化后的值，0-1 范围）
        score = prop.intensity_max if prop.intensity_max is not None else 1.0
        
        targets.append((cx, cy, float(score)))
    
    return targets


def get_image_size(image_path: str) -> Tuple[int, int]:
    """获取图像尺寸"""
    with Image.open(image_path) as img:
        return img.size  # (width, height)


def find_matching_files(gt_dir: str, pred_dir: str, image_dir: str) -> List[Dict]:
    """
    找到匹配的文件
    
    Returns:
        [{'id': image_id, 'gt_mask': path, 'pred_mask': path, 'image': path}, ...]
    """
    # 支持的图像格式
    img_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    
    # 获取所有文件（支持去除 _mask 后缀）
    def get_files(directory, strip_mask_suffix=False):
        files = {}
        if os.path.exists(directory):
            for f in os.listdir(directory):
                name, ext = os.path.splitext(f)
                if ext.lower() in img_exts:
                    # 去除常见的 mask 后缀（如 _mask, _gt, _label）
                    if strip_mask_suffix:
                        for suffix in ['_mask', '_gt', '_label', '_GT', '_Mask', '_Label']:
                            if name.endswith(suffix):
                                name = name[:-len(suffix)]
                                break
                    files[name] = os.path.join(directory, f)
        return files
    
    # GT masks 可能有 _mask 后缀，需要去除
    gt_files = get_files(gt_dir, strip_mask_suffix=True)
    pred_files = get_files(pred_dir, strip_mask_suffix=False)
    image_files = get_files(image_dir, strip_mask_suffix=False)
    
    # 以预测 mask 为基准（GT 和 images 可能包含 train+test，但预测只有 test）
    all_ids = sorted(pred_files.keys())
    
    matches = []
    for img_id in all_ids:
        matches.append({
            'id': img_id,
            'gt_mask': gt_files.get(img_id),
            'pred_mask': pred_files.get(img_id),
            'image': image_files.get(img_id)
        })
    
    return matches


# ============================================================
# 主评估函数
# ============================================================

def evaluate_from_masks(
    gt_mask_dir: str,
    pred_mask_dir: str,
    image_dir: str,
    tp_distance: float = 5.0,
    pdfa_thresholds: Optional[List[float]] = None,
    use_weighted_centroid: bool = True,
    default_size: Tuple[int, int] = (640, 512)
) -> Dict:
    """
    从 mask 目录计算评价指标
    
    Args:
        gt_mask_dir: GT mask 目录
        pred_mask_dir: 预测 mask 目录
        image_dir: 原始图像目录
        tp_distance: TP 距离阈值
        pdfa_thresholds: 置信度阈值列表
        use_weighted_centroid: 是否使用加权质心
        default_size: 默认图像尺寸 (width, height)
    
    Returns:
        包含 metrics, pdfa_table, bad_image_ids 的字典
    """
    # 默认阈值
    if pdfa_thresholds is None:
        pdfa_thresholds = [round(i * 0.1, 1) for i in range(1, 11)]
    
    # 找到匹配的文件
    file_matches = find_matching_files(gt_mask_dir, pred_mask_dir, image_dir)
    
    if not file_matches:
        print("Warning: No matching files found!")
        return {'metrics': {}, 'pdfa_table': {}, 'per_image': [], 'bad_image_ids': []}
    
    # 存储结果
    per_image_results = []
    match_cache = []
    total_pixels = 0.0
    
    # 基础指标累加
    total_tp, total_fp, total_fn = 0, 0, 0
    
    print(f"Processing {len(file_matches)} images...")
    
    for item in file_matches:
        img_id = item['id']
        
        # 提取 GT 目标
        gt_targets = []
        if item['gt_mask'] and os.path.exists(item['gt_mask']):
            gt_targets = extract_targets_from_mask(item['gt_mask'], use_weighted_centroid)
        
        # 提取预测目标
        pred_targets = []
        if item['pred_mask'] and os.path.exists(item['pred_mask']):
            pred_targets = extract_targets_from_mask(item['pred_mask'], use_weighted_centroid)
        
        # 提取 GT 坐标 (x, y)
        gt_coords = [(x, y) for x, y, _ in gt_targets]
        
        # 匹配
        tp, fp, fn, matched_indices = match_keypoints_gt_driven(
            pred_targets, gt_coords, tp_distance
        )
        
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # 记录结果
        per_image_results.append({
            'image_id': img_id,
            'tp': tp, 'fp': fp, 'fn': fn,
            'num_gt': len(gt_coords),
            'num_pred': len(pred_targets)
        })
        
        # 缓存匹配状态
        scores = [s for _, _, s in pred_targets]
        match_cache.append({
            'matched_indices': matched_indices,
            'scores': scores,
            'num_gt': len(gt_coords),
            'num_pred': len(pred_targets)
        })
        
        # 累加像素数
        if item['image'] and os.path.exists(item['image']):
            w, h = get_image_size(item['image'])
        elif item['gt_mask'] and os.path.exists(item['gt_mask']):
            w, h = get_image_size(item['gt_mask'])
        else:
            w, h = default_size
        total_pixels += w * h
    
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
        'image_count': len(file_matches)
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


# ============================================================
# 输出与保存（与 eval_from_json.py 保持一致）
# ============================================================

def print_pdfa_table(pdfa_table: Dict):
    """打印 PD_FA 阈值表"""
    print("-" * 120)
    print(f"{'Thresh':<8} {'TP':>8} {'FP':>8} {'FN':>8} "
          f"{'Precision':>10} {'Recall':>10} {'F1':>10} {'PD':>10} {'FA (e-08)':>14}")
    print("-" * 120)
    
    for i, t in enumerate(pdfa_table['thresholds']):
        # FA 转换为 e-08 数量级，保留 4 位小数
        fa_e8 = pdfa_table['FA'][i] * 1e8
        print(f"{t:<8.2f} {pdfa_table['TP'][i]:>8} {pdfa_table['FP'][i]:>8} "
              f"{pdfa_table['FN'][i]:>8} {pdfa_table['Precision'][i]:>10.6f} "
              f"{pdfa_table['Recall'][i]:>10.6f} {pdfa_table['F1'][i]:>10.6f} "
              f"{pdfa_table['PD'][i]:>10.6f} {fa_e8:>14.4f}")
    
    print("-" * 120)


def save_results(results: Dict, output_path: str, tp_distance: float, thresholds: List[float], 
                 gt_dir: str, pred_dir: str):
    """保存结果到文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("IRST Metrics Evaluation Results (from Masks)\n")
        f.write("=" * 120 + "\n\n")
        
        # 配置
        f.write("【Configuration】\n")
        f.write("-" * 80 + "\n")
        f.write(f"GT Mask Directory: {gt_dir}\n")
        f.write(f"Pred Mask Directory: {pred_dir}\n")
        f.write(f"TP Distance Threshold: {tp_distance}\n")
        f.write(f"Thresholds: {thresholds}\n")
        f.write(f"Target Extraction: 8-connectivity clustering (skimage.measure.label)\n\n")
        
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
                f"{'Precision':>12} {'Recall':>12} {'F1':>12} {'FA (e-08)':>15}\n")
        f.write("-" * 120 + "\n")
        
        for i, t in enumerate(table['thresholds']):
            # FA 转换为 e-08 数量级，保留 4 位小数
            fa_e8 = table['FA'][i] * 1e8
            f.write(f"{t:<10.2f} {table['TP'][i]:>8} {table['FP'][i]:>8} "
                    f"{table['FN'][i]:>8} {table['Precision'][i]:>12.6f} "
                    f"{table['Recall'][i]:>12.6f} {table['F1'][i]:>12.6f} "
                    f"{fa_e8:>15.4f}\n")
        f.write("\n")
        
        # 错误图像
        bad_ids = results['bad_image_ids']
        f.write(f"【Error Images】({len(bad_ids)} total)\n")
        f.write("-" * 80 + "\n")
        if bad_ids:
            f.write(", ".join(bad_ids) + "\n")


# ============================================================
# 主函数
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description='红外小目标检测指标评估工具（Mask 版本）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--gt-masks', required=True, help='GT mask 目录')
    parser.add_argument('--pred-masks', required=True, help='预测 mask 目录')
    parser.add_argument('--images', required=True, help='原始图像目录（用于获取尺寸）')
    parser.add_argument('--tp-distance', type=float, default=5.0, help='TP 距离阈值 (默认: 5.0)')
    parser.add_argument('--thresholds', type=float, nargs='*', default=None,
                        help='阈值列表 (默认: 0.1-1.0 间隔 0.1)')
    parser.add_argument('--output', type=str, default=None, help='结果输出文件路径')
    parser.add_argument('--no-weighted-centroid', action='store_true',
                        help='使用几何质心而非加权质心')
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 检查目录
    if not os.path.exists(args.gt_masks):
        print(f"Error: GT mask directory not found: {args.gt_masks}")
        sys.exit(1)
    if not os.path.exists(args.pred_masks):
        print(f"Error: Prediction mask directory not found: {args.pred_masks}")
        sys.exit(1)
    if not os.path.exists(args.images):
        print(f"Error: Image directory not found: {args.images}")
        sys.exit(1)
    
    # 设置阈值
    thresholds = args.thresholds or [round(i * 0.1, 1) for i in range(1, 11)]
    
    # 打印头部
    print("=" * 120)
    print("IRST Metrics Evaluation from Masks")
    print("=" * 120)
    print(f"GT masks: {args.gt_masks}")
    print(f"Pred masks: {args.pred_masks}")
    print(f"Images: {args.images}")
    print(f"TP distance: {args.tp_distance}")
    print(f"Thresholds: {thresholds}")
    print(f"Centroid: {'geometric' if args.no_weighted_centroid else 'weighted'}")
    print("-" * 120)
    
    # 评估
    results = evaluate_from_masks(
        args.gt_masks, args.pred_masks, args.images,
        tp_distance=args.tp_distance,
        pdfa_thresholds=thresholds,
        use_weighted_centroid=not args.no_weighted_centroid
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
    if f1_list:
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
        save_results(results, args.output, args.tp_distance, thresholds,
                     args.gt_masks, args.pred_masks)
        print(f"\nResults saved to: {args.output}")
    
    return results


if __name__ == '__main__':
    main()
