

import os
import json
import cv2
import numpy as np
from skimage import measure
import argparse


# ========================================
#   核心逻辑：从mask -> 提取点+框+区域 -> Json
# ========================================

def extract_points_and_boxes_from_mask(mask_bin):
    """
    输入：二值 mask (0/1)
    输出：
        pts   : [(cx, cy), ...]
        boxes : [(x, y, w, h), ...]   # x,y 是左上角，来自 regionprops.bbox
        areas : [area1, area2, ...]
    """
    labeled = measure.label(mask_bin, connectivity=2)
    props = measure.regionprops(labeled)

    pts = []
    boxes = []
    areas = []

    for region in props:
        # 质心（注意：regionprops 返回 (row, col) = (y, x)）
        cy, cx = region.centroid
        pts.append((float(cx), float(cy)))

        # bbox: (min_row, min_col, max_row, max_col)
        minr, minc, maxr, maxc = region.bbox
        x = float(minc)
        y = float(minr)
        w = float(maxc - minc)
        h = float(maxr - minr)
        boxes.append((x, y, w, h))

        # 区域面积（像素个数）
        areas.append(float(region.area))

    # ⚠ 无前景目标时，返回空列表
    return pts, boxes, areas


# ===========================
#   绘制图像：质心细点
# ===========================
def draw_point(img, cx, cy, r=2):
    """
    在质心位置画带白色描边的实心圆，样式与 val2026.py draw_keypoints 一致
    fill=(240,2,127) RGB -> BGR=(127,2,240), outline=white
    """
    cv2.circle(img, (cx, cy), r, (127, 2, 240), -1, lineType=cv2.LINE_AA)
    cv2.circle(img, (cx, cy), r, (255, 255, 255), 1, lineType=cv2.LINE_AA)


# ===========================
#   可视化函数
# ===========================
def save_comparison_figure(
        fname, mask_gray, pts, boxes,
        image_dir, out_vis_dir, save_vis=False
):
    """
    生成 2×2 对比图：
    [ 原图 | mask ]
    [ mask+点+框 | 原图+点+框 ]

    当 save_vis=False 时，直接 return，不做任何图像处理。
    """
    if not save_vis:
        return

    img_path = os.path.join(image_dir, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("WARNING: cannot read image:", img_path)
        return

    # 尺寸对齐
    if img.shape != mask_gray.shape:
        img = cv2.resize(img, (mask_gray.shape[1], mask_gray.shape[0]))

    img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask_color = cv2.cvtColor(mask_gray, cv2.COLOR_GRAY2BGR)

    # 用于绘制点+框
    mask_with = mask_color.copy()
    img_with = img_color.copy()

    # ====== 画每个目标的质心细点 + 外接细框 ======
    for (pt, box) in zip(pts, boxes):
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        draw_point(mask_with, cx, cy)
        draw_point(img_with,  cx, cy)

        x, y, w, h = box
        if w > 1 and h > 1:
            p1 = (int(round(x)),         int(round(y)))
            p2 = (int(round(x + w)),     int(round(y + h)))
            # mask 上黄色细框
            cv2.rectangle(mask_with, p1, p2, (0, 255, 255), 1)
            # 原图上蓝色细框
            cv2.rectangle(img_with,  p1, p2, (255, 0, 0), 1)

    # --- 第一行：原图 + mask ---
    row1 = np.concatenate([img_color, mask_color], axis=1)

    # --- 第二行：mask+点+框 + 原图+点+框 ---
    row2 = np.concatenate([img_with, mask_with], axis=1)

    # --- 最终 2×2 拼接 ---
    concat = np.concatenate([row1, row2], axis=0)

    os.makedirs(out_vis_dir, exist_ok=True)
    cv2.imwrite(os.path.join(out_vis_dir, fname), concat)

    # ====== 只画质心细点（不画框）======
    img_point_only = img_color.copy()
    for pt in pts:
        cx, cy = int(round(pt[0])), int(round(pt[1]))
        draw_point(img_point_only, cx, cy)
    os.makedirs(out_vis_dir + "single", exist_ok=True)
    cv2.imwrite(os.path.join(out_vis_dir + "single", fname), img_point_only)



    




# ===========================
#   主构建函数（可直接 import 使用）
# ===========================
def build_coco_from_masks(mask_dir,
                          image_dir,
                          out_json,
                          out_vis_dir=None,
                          save_vis=False):
    masks = sorted([
        f for f in os.listdir(mask_dir)
        if f.lower().endswith(( ".jpg", ".png", ".bmp", ".tif", ".tiff"))
    ])

    images = []
    annotations = []
    ann_id = 1

    total = len(masks)

    for idx, fname in enumerate(masks, start=1):
        # ====== 终端输出进度提示 ======
        print(f"[{idx}/{total}] Processing: {fname}")

        path = os.path.join(mask_dir, fname)
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if mask is None:
            print("WARNING: cannot read mask:", path)
            continue

        mask_bin = (mask > 0).astype(np.uint8)
        H, W = mask_bin.shape

        # 提取质心、bbox、area（来自 regionprops）
        pts, boxes, areas = extract_points_and_boxes_from_mask(mask_bin)
        num_points = len(pts)

        image_id = os.path.splitext(fname)[0]

        # ====== images ======
        images.append({
            "file_name": fname,
            "height": H,
            "width": W,
            "id": image_id
        })

        # ====== annotations ======
        keypoints = []
        for (x, y) in pts:
            keypoints.extend([x, y, 1])  # v=1 表示标注存在/可见

        # bbox：多个目标 -> 嵌套列表 [[x1,y1,w1,h1], ...]；无目标 -> []
        if len(boxes) > 0:
            bbox_field = [[float(x), float(y), float(w), float(h)] for (x, y, w, h) in boxes]
        else:
            bbox_field = []

        # area：多个目标 -> [a1, a2, ...]；无目标 -> []
        if len(areas) > 0:
            area_field = [float(a) for a in areas]
        else:
            area_field = []

        # ✅ 无论有没有目标，都会写一条 annotation（bbox/area/keypoints 可以是空）
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "bbox": bbox_field,
            "area": area_field,
            "category_id": 1,
            "num_keypoints": num_points,
            "keypoints": keypoints,
        })

        ann_id += 1

        # ====== 可视化（完全可关闭） ======
        if save_vis and out_vis_dir is not None:
            save_comparison_figure(
                fname, mask, pts, boxes,
                image_dir, out_vis_dir, save_vis=save_vis
            )

    # ====== 输出 JSON ======
    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "target"}]
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f, indent=2)

    print("JSON saved to:", out_json)
    if save_vis and out_vis_dir is not None:
        print("Visualization saved to:", out_vis_dir)


# ===========================
#   命令行入口
# ===========================
if __name__ == "__main__":

    build_coco_from_masks(
        mask_dir=r"N:\AcademicResearchs\Experiments\LiTENetExp\SIRST4\PredImg_Masks\SCTrans",
        image_dir=r"N:\AcademicResearchs\Datasets\IRSingleFrame_Dataset\SIRST4\images",
        out_json=r"N:\AcademicResearchs\Experiments\LiTENetExp\SIRST4\Annotations\SCTrans_annotations.json",
        out_vis_dir=r"N:\AcademicResearchs\Experiments\LiTENetExp\SIRST4\PredImg_Point\SCTrans_vis",
        save_vis=True
    )

