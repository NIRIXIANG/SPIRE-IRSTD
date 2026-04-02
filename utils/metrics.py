import os
import numpy as np


def detect_dataset_format(root):
    """
    自动检测数据集目录格式
    返回: 'format_a' (原格式: train/train_images/) 或 'format_b' (新格式: images/ + img_idx/)
    """
    if root is None:
        return None

    images_dir = os.path.join(root, "images")
    img_idx_dir = os.path.join(root, "img_idx")
    unified_anno = os.path.join(root, "annotations", "annotations.json")

    if os.path.exists(images_dir) and os.path.exists(img_idx_dir) and os.path.exists(unified_anno):
        return 'format_b'

    train_dir = os.path.join(root, "train", "train_images")
    test_dir = os.path.join(root, "test", "test_images")

    if os.path.exists(train_dir) or os.path.exists(test_dir):
        return 'format_a'

    return 'format_a'


def get_gt_keypoints(gt_data, image_id):
    """
    从GT数据中获取指定图像的关键点坐标
    """
    keypoints = []
    for annotation in gt_data['annotations']:
        if annotation['image_id'] == image_id:
            keypoints_data = annotation['keypoints']
            for i in range(0, len(keypoints_data), 3):
                keypoints.append(keypoints_data[i:i + 2])
    return np.array(keypoints)


def calculate_metrics(pred_keypoints, gt_keypoints, tp_distance):
    """
    GT驱动的一对一匹配指标计算
    - TP = 被成功匹配到的 GT 数量
    - FN = GT 总数 - TP
    - FP = 未被匹配到的预测点数量
    """
    pred_keypoints = np.asarray(pred_keypoints, dtype=np.float32)
    gt_keypoints = np.asarray(gt_keypoints, dtype=np.float32)

    if pred_keypoints.ndim == 1:
        pred_keypoints = pred_keypoints[None, :]

    valid_pred_mask = ~np.logical_and(
        pred_keypoints[:, 0] == 0,
        pred_keypoints[:, 1] == 0
    )
    pred_keypoints = pred_keypoints[valid_pred_mask]

    num_gt = gt_keypoints.shape[0]
    num_pred = pred_keypoints.shape[0]

    if num_gt == 0:
        return 0, int(num_pred), 0
    if num_pred == 0:
        return 0, 0, int(num_gt)

    tp = 0
    pred_used = np.zeros(num_pred, dtype=bool)

    for gi, gt in enumerate(gt_keypoints):
        dists = np.linalg.norm(pred_keypoints - gt[None, :], axis=1)
        candidate_idx = np.where(~pred_used)[0]
        if candidate_idx.size == 0:
            break
        cand_dists = dists[candidate_idx]
        min_arg = np.argmin(cand_dists)
        min_idx = candidate_idx[min_arg]
        min_dist = cand_dists[min_arg]

        if min_dist <= tp_distance:
            tp += 1
            pred_used[min_idx] = True

    fp = int(np.sum(~pred_used))
    fn = int(num_gt - tp)

    return tp, fp, fn


def compute_prf1(total_tp, total_fp, total_fn):
    """计算 Precision, Recall, F1 Score"""
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score
