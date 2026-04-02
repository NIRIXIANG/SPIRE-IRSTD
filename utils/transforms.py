"""SPIRE 数据增广与热图/解码：仅保留训练、DDP 训练与 evaluate 所需接口。"""
import math
import os
import random
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import functional as F


def affine_points(pt, t):
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt = np.concatenate([pt, ones], axis=1).T
    new_pt = np.dot(t, pt)
    return new_pt.T


def nrx_affine_points(pt, t):
    zero_positions = np.where(pt == 0)
    ones = np.ones((pt.shape[0], 1), dtype=float)
    pt_homogeneous = np.concatenate([pt, ones], axis=1).T
    new_pt_homogeneous = np.dot(t, pt_homogeneous)
    new_pt = new_pt_homogeneous.T[:, :2]
    new_pt[zero_positions] = 0
    return new_pt


def nrx_heatmap_nms(batch_heatmaps):
    heatmap = batch_heatmaps.clone().cpu()
    pool = torch.nn.MaxPool2d(3, stride=1, padding=1)
    maxm = pool(heatmap)
    maxm = torch.eq(maxm, heatmap).float()
    heatmap = heatmap * maxm
    return heatmap


def nrx_get_max_preds(batch_heatmaps, num_targets, threshold, value_range):
    preds = torch.zeros((batch_heatmaps.shape[0], num_targets, 2), dtype=torch.float32)
    maxvals = torch.zeros((batch_heatmaps.shape[0], num_targets, 1), dtype=torch.float32)

    for batch_idx in range(batch_heatmaps.shape[0]):
        heatmaps = batch_heatmaps[batch_idx]
        target_points = []

        for channel_idx in range(heatmaps.shape[0]):
            heatmap = heatmaps[channel_idx]
            above_threshold = torch.where(heatmap > threshold)
            y_coords, x_coords = above_threshold[0], above_threshold[1]

            for i in range(len(y_coords)):
                y, x = y_coords[i], x_coords[i]
                confidence = heatmap[y, x]
                target_points.append((x, y, confidence))

        target_points.sort(key=lambda x: x[2], reverse=True)
        selected_points = []
        for point in target_points:
            x, y, confidence = point
            valid = True
            for selected_point in selected_points:
                sx, sy, _ = selected_point
                max_value = max(heatmap[y, x], heatmap[sy, sx])
                min_value = min(heatmap[y, x], heatmap[sy, sx])
                if max_value - min_value > value_range * max_value:
                    valid = False
                    break

            if valid:
                selected_points.append(point)
                if len(selected_points) == num_targets:
                    break

        for i, point in enumerate(selected_points):
            x, y, confidence = point
            preds[batch_idx, i, 0] = x
            preds[batch_idx, i, 1] = y
            maxvals[batch_idx, i, 0] = confidence

    return preds, maxvals


def nrx_get_final_preds(
    batch_heatmaps: torch.Tensor,
    trans: list = None,
    post_processing: bool = False,
    output_dir: str = None,
    img_name: str = None,
    threshold: float = None,
    value_range: float = None,
    max_num_targets: int = None,
    save_heatmap: bool = False,
):
    assert trans is not None

    nms_batch_heatmaps = nrx_heatmap_nms(batch_heatmaps)

    if save_heatmap:
        plt.clf()
        cax = plt.imshow(nms_batch_heatmaps[0, 0].numpy(), cmap='viridis', interpolation='nearest')
        plt.colorbar(cax, fraction=0.046, pad=0.04)
        nms_heatmap_save_path = os.path.join(output_dir, f"{img_name}nmsheatmap.png")
        plt.savefig(nms_heatmap_save_path, format='png', dpi=500)

    coords, maxvals = nrx_get_max_preds(
        nms_batch_heatmaps,
        num_targets=max_num_targets,
        threshold=threshold,
        value_range=value_range,
    )

    heatmap_height = batch_heatmaps.shape[2]
    heatmap_width = batch_heatmaps.shape[3]

    if save_heatmap:
        batch_index = 0
        channel_index = 0
        heatmap = batch_heatmaps.clone().cpu()[batch_index, channel_index, :, :]
        plt.clf()
        cax = plt.imshow(heatmap, cmap='viridis', interpolation='nearest')
        plt.colorbar(cax, fraction=0.046, pad=0.04)
        heatmap_save_path = os.path.join(output_dir, f"{img_name}heatmap.png")
        plt.savefig(heatmap_save_path, format='png', dpi=500)

    if post_processing:
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                hm = batch_heatmaps[n][0]
                if (coords[n][p][0] != 0) and (coords[n][p][1] != 0):
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = torch.tensor(
                            [
                                hm[py][px + 1] - hm[py][px - 1],
                                hm[py + 1][px] - hm[py - 1][px],
                            ]
                        )
                        coords[n][p] += torch.sign(diff) * .25
                else:
                    continue

    preds = coords.clone().cpu().numpy()
    for i in range(coords.shape[0]):
        preds[i] = nrx_affine_points(preds[i], trans[i])

    return preds, maxvals.cpu().numpy()


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target


class Normalize(object):
    def __init__(self, mean=None, std=None):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


class AffineTransform(object):
    """保持宽高比的缩放 + padding；可选随机缩放与旋转。"""

    def __init__(
        self,
        scale: Tuple[float, float] = None,
        rotation: Tuple[int, int] = None,
        fixed_size: Tuple[int, int] = (512, 512),
    ):
        self.scale = scale
        self.rotation = rotation
        self.fixed_size = fixed_size

    def __call__(self, img, target):
        h, w = img.shape[0], img.shape[1]
        dst_h, dst_w = self.fixed_size[0], self.fixed_size[1]

        scale_ratio = min(dst_h / h, dst_w / w)
        if self.scale is not None:
            scale_factor = random.uniform(*self.scale)
            scale_ratio *= scale_factor

        new_h = h * scale_ratio
        new_w = w * scale_ratio
        pad_h = (dst_h - new_h) / 2
        pad_w = (dst_w - new_w) / 2

        src = np.array([[0, 0], [w - 1, 0], [0, h - 1]], dtype=np.float32)
        dst = np.array(
            [
                [pad_w, pad_h],
                [dst_w - pad_w - 1, pad_h],
                [pad_w, dst_h - pad_h - 1],
            ],
            dtype=np.float32,
        )

        if self.rotation is not None:
            angle = random.randint(*self.rotation)
            angle_rad = angle / 180 * math.pi
            center = np.array([(dst_w - 1) / 2, (dst_h - 1) / 2])
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            for i in range(3):
                offset = dst[i] - center
                rotated_offset = np.array([
                    offset[0] * cos_a - offset[1] * sin_a,
                    offset[0] * sin_a + offset[1] * cos_a,
                ])
                dst[i] = center + rotated_offset

        trans = cv2.getAffineTransform(src, dst)
        dst_heatmap = dst / 4
        reverse_trans = cv2.getAffineTransform(dst_heatmap, src)

        resize_img = cv2.warpAffine(
            img,
            trans,
            (dst_w, dst_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        if "keypoints" in target:
            kps = target["keypoints"]
            mask = np.logical_and(kps[:, 0] != 0, kps[:, 1] != 0)
            kps[mask] = affine_points(kps[mask], trans)
            for i in range(len(kps)):
                if mask[i]:
                    x, y = kps[i]
                    if x < 0 or x >= dst_w or y < 0 or y >= dst_h:
                        kps[i] = [0, 0]
            target["keypoints"] = kps

        target["trans"] = trans
        target["reverse_trans"] = reverse_trans
        return resize_img, target


def _make_gaussian_kernel(gaussian_sigma: int):
    kernel_radius = gaussian_sigma * 3
    kernel_size = 2 * kernel_radius + 1
    kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
    x_center = y_center = kernel_size // 2
    for x in range(kernel_size):
        for y in range(kernel_size):
            kernel[y, x] = np.exp(
                -((x - x_center) ** 2 + (y - y_center) ** 2) / (2 * gaussian_sigma ** 2)
            )
    return kernel, kernel_radius


class nrxKeypointToHeatMap_oneMapwithManyPoints(object):
    

    def __init__(
        self,
        heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
        gaussian_sigma: int = 2,
        keypoints_weights=None,
    ):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel, self.kernel_radius = _make_gaussian_kernel(gaussian_sigma)
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)

        heatmap = np.zeros((1, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(int)
        for kp_id in range(num_kps):
            x, y = heatmap_kps[kp_id]
            if (x, y) == (0, 0):
                continue
            ul = [x - self.kernel_radius, y - self.kernel_radius]
            br = [x + self.kernel_radius, y + self.kernel_radius]
            if (
                ul[0] > self.heatmap_hw[1] - 1
                or ul[1] > self.heatmap_hw[0] - 1
                or br[0] < 0
                or br[1] < 0
            ):
                kps_weights[kp_id] = 0
                continue

            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            for i in range(g_y[0], g_y[1] + 1):
                for j in range(g_x[0], g_x[1] + 1):
                    heatmap_y = img_y[0] + i - g_y[0]
                    heatmap_x = img_x[0] + j - g_x[0]
                    if heatmap[0, heatmap_y, heatmap_x] > 0:
                        heatmap[0, heatmap_y, heatmap_x] = (
                            heatmap[0, heatmap_y, heatmap_x] + self.kernel[i, j]
                        ) / 2
                    else:
                        heatmap[0, heatmap_y, heatmap_x] = self.kernel[i, j]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)
        return image, target


class nrxKeypointToHeatMap_targetEnhance(object):
    

    def __init__(
        self,
        heatmap_hw: Tuple[int, int] = (256 // 4, 192 // 4),
        gaussian_sigma: int = 2,
        keypoints_weights=None,
    ):
        self.heatmap_hw = heatmap_hw
        self.sigma = gaussian_sigma
        self.kernel, self.kernel_radius = _make_gaussian_kernel(gaussian_sigma)
        self.use_kps_weights = False if keypoints_weights is None else True
        self.kps_weights = keypoints_weights

    def correct_kps(self, img, kp, neighborhood_size=10):
        corrected_kps = []
        x, y = int(kp[0]), int(kp[1])
        half_size = neighborhood_size // 2
        ymin = max(0, y - half_size)
        ymax = min(img.shape[0], y + half_size + 1)
        xmin = max(0, x - half_size)
        xmax = min(img.shape[1], x + half_size + 1)
        neighborhood = img[ymin:ymax, xmin:xmax]

        if neighborhood.size == 0:
            print(f"kps x:{x},y:{y}")
            print(
                f"Warning: Neighborhood size is invalid for keypoint at ({x}, {y}). Keeping original coordinates."
            )
            corrected_kps.append([x, y])
            return corrected_kps

        max_value_index = np.unravel_index(np.argmax(neighborhood), neighborhood.shape)
        corrected_x = xmin + max_value_index[1]
        corrected_y = ymin + max_value_index[0]
        corrected_kps.append([corrected_x, corrected_y])
        return corrected_kps

    def getCropImg(self, img, kp):
        centroid_label_x = kp[0][0]
        centroid_label_y = kp[0][1]
        Ymin_f = int(max(0, centroid_label_y - self.kernel_radius))
        Ymax_f = int(min(img.shape[0], centroid_label_y + self.kernel_radius + 1))
        Xmin_f = int(max(0, centroid_label_x - self.kernel_radius))
        Xmax_f = int(min(img.shape[1], centroid_label_x + self.kernel_radius + 1))

        crop_height = Ymax_f - Ymin_f
        crop_width = Xmax_f - Xmin_f

        if crop_height < (2 * self.kernel_radius + 1):
            if Ymin_f > 0:
                Ymin_f = Ymax_f - (2 * self.kernel_radius + 1)
            else:
                Ymax_f = Ymin_f + (2 * self.kernel_radius + 1)

        if crop_width < (2 * self.kernel_radius + 1):
            if Xmin_f > 0:
                Xmin_f = Xmax_f - (2 * self.kernel_radius + 1)
            else:
                Xmax_f = Xmin_f + (2 * self.kernel_radius + 1)

        crop_image = img[Ymin_f:Ymax_f, Xmin_f:Xmax_f]
        return crop_image

    def process_image(self, crop_image, min_val=0, max_val=255):
        result_img = crop_image.copy()
        target_values = crop_image.flatten()
        min_target_value = np.min(target_values)
        max_target_value = np.max(target_values)

        if min_target_value == max_target_value:
            print("Warning: min and max values are the same. Skipping mapping.")
            return result_img

        mapped_values = self.nonlinear_mapping(
            target_values, min_target_value, max_target_value, min_val, max_val
        )
        result_img = mapped_values.reshape(crop_image.shape)
        return result_img

    def nonlinear_mapping(self, values, min_target_value, max_target_value, min_val, max_val):
        if max_target_value > min_target_value:
            mapped_values = np.interp(values, (min_target_value, max_target_value), (min_val, max_val))
        else:
            mapped_values = values
        mapped_values[mapped_values == max_val] = 255
        return mapped_values

    def normalize_to_heatmap(self, processed_image):
        gray_image = np.dot(processed_image[..., :3], [0.2989, 0.5870, 0.1140])
        normalized_image = gray_image / 255.0
        return normalized_image

    def restore_and_show_heatmap(self, normalized_image):
        restored_image = np.uint8(normalized_image * 255)
        plt.imshow(restored_image, cmap='hot')
        plt.colorbar()
        plt.title("Restored Heatmap (0-255)")
        plt.axis('off')
        plt.show()

    def __call__(self, image, target):
        kps = target["keypoints"]
        num_kps = kps.shape[0]
        kps_weights = np.ones((num_kps,), dtype=np.float32)

        heatmap = np.zeros((1, self.heatmap_hw[0], self.heatmap_hw[1]), dtype=np.float32)
        heatmap_kps = (kps / 4 + 0.5).astype(int)
        for kp_id in range(num_kps):
            x, y = heatmap_kps[kp_id]
            if (x, y) == (0, 0):
                continue
            ul = [x - self.kernel_radius, y - self.kernel_radius]
            br = [x + self.kernel_radius, y + self.kernel_radius]
            if (
                ul[0] > self.heatmap_hw[1] - 1
                or ul[1] > self.heatmap_hw[0] - 1
                or br[0] < 0
                or br[1] < 0
            ):
                kps_weights[kp_id] = 0
                continue

            correct_kps = self.correct_kps(image, kps[kp_id])
            crop_image = self.getCropImg(image, correct_kps)
            processed_image = self.process_image(crop_image, min_val=0, max_val=255)
            normalized_patch = self.normalize_to_heatmap(processed_image)
            gaussian_patch = self.kernel
            result_patch = gaussian_patch * normalized_patch
            min_val = result_patch.min()
            max_val = result_patch.max()
            result_patch = (result_patch - min_val) / (max_val - min_val)

            if min_val == max_val:
                print("Warning: min and max values are the same. Skipping mapping.")
                result_patch.fill(0)
            else:
                result_patch = (result_patch - min_val) / (max_val - min_val)

            g_x = (max(0, -ul[0]), min(br[0], self.heatmap_hw[1] - 1) - ul[0])
            g_y = (max(0, -ul[1]), min(br[1], self.heatmap_hw[0] - 1) - ul[1])
            img_x = (max(0, ul[0]), min(br[0], self.heatmap_hw[1] - 1))
            img_y = (max(0, ul[1]), min(br[1], self.heatmap_hw[0] - 1))

            for i in range(g_y[0], g_y[1] + 1):
                for j in range(g_x[0], g_x[1] + 1):
                    heatmap_y = img_y[0] + i - g_y[0]
                    heatmap_x = img_x[0] + j - g_x[0]
                    if heatmap[0, heatmap_y, heatmap_x] > 0:
                        heatmap[0, heatmap_y, heatmap_x] = (
                            heatmap[0, heatmap_y, heatmap_x] + result_patch[i, j]
                        ) / 2
                    else:
                        heatmap[0, heatmap_y, heatmap_x] = result_patch[i, j]

        if self.use_kps_weights:
            kps_weights = np.multiply(kps_weights, self.kps_weights)

        target["heatmap"] = torch.as_tensor(heatmap, dtype=torch.float32)
        target["kps_weights"] = torch.as_tensor(kps_weights, dtype=torch.float32)
        return image, target
