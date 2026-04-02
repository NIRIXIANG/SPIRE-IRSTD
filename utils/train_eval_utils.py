import math
import sys
import numpy as np
import torch
import json
import os

from . import transforms
from . import distributed_utils as dist_utils
from .loss import KpLoss
from .metrics import detect_dataset_format, get_gt_keypoints, calculate_metrics, compute_prf1


def train_one_epoch(model, optimizer, data_loader, device, epoch,
                    print_freq=50, warmup=False, scaler=None):
    model.train()
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', dist_utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0 and warmup is True:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = dist_utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    mse = KpLoss()
    mloss = torch.zeros(1).to(device)

    for i, [images, targets] in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = images.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            results = model(images)
            losses = mse(results, targets)

        loss_dict_reduced = dist_utils.reduce_dict({"losses": losses})
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()
        mloss = (mloss * i + loss_value) / (i + 1)

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced)
        now_lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=now_lr)

    return mloss, now_lr


def validate_one_epoch(model, data_loader, device, scaler=None):
    model.eval()
    metric_logger = dist_utils.MetricLogger(delimiter="  ")
    mse = KpLoss()

    total_loss = 0
    num_loss_cal = 0
    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 50, "Val loss"):
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                loss = mse(outputs, targets)
                num_loss_cal += 1
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)
    print(f"Validation Loss: {avg_loss:.4f}")
    print(f"calculate loss: {num_loss_cal} times")
    return avg_loss


def nrx_evaluate(model, data_loader, device, scaler, threshold, value_range,
                 max_num_targets, output_dir, data_path, tp_distance):
    model.eval()
    metric_logger = dist_utils.MetricLogger(delimiter="  ")

    total_tp = 0
    total_fp = 0
    total_fn = 0

    assert os.path.exists(data_path), "path '{}' does not exist.".format(data_path)

    dataset_format = detect_dataset_format(data_path)
    if dataset_format == 'format_b':
        gt_json_path = os.path.join(data_path, "annotations", "annotations.json")
    else:
        gt_json_path = os.path.join(data_path, "test", "annotations", "test.json")

    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    with torch.no_grad():
        for images, targets in metric_logger.log_every(data_loader, 50, "Evaluate"):
            images = images.to(device)
            with torch.cuda.amp.autocast(enabled=scaler is not None):
                outputs = model(images)
                for i in range(outputs.shape[0]):
                    output = outputs[i].unsqueeze(0)
                    target = targets[i]
                    keypoints, scores = transforms.nrx_get_final_preds(
                        output, [target["reverse_trans"]],
                        True, output_dir, None, threshold, value_range, max_num_targets
                    )
                    keypoints = np.squeeze(keypoints)
                    if keypoints.ndim == 1:
                        keypoints = np.array([keypoints])

                    image_id = target["image_id"]
                    gt_keypoints = get_gt_keypoints(gt_data, image_id)

                    tp, fp, fn = calculate_metrics(keypoints, gt_keypoints, tp_distance)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

    precision, recall, f1_score = compute_prf1(total_tp, total_fp, total_fn)

    print(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

    return total_tp, total_fp, total_fn, precision, recall, f1_score
