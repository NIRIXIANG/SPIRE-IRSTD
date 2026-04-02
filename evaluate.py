import argparse
import time
import os
import json
import importlib.util
import sys

import torch
import numpy as np
import cv2
from PIL import Image

from model import SPIRENet
from utils.experiment_utils import infer_experiment_name_from_weights, resolve_output_subdir
from utils import transforms
from utils.metrics import detect_dataset_format, get_gt_keypoints, calculate_metrics, compute_prf1
from utils.draw_utils import draw_keypoints

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")

parser = argparse.ArgumentParser(description="PyTorch SPIRE evaluate")
parser.add_argument("--weights_path", default=r"N:\AcademicResearchs\MyProjects\LiTENet_New\20260302_SPIRE_uavb_batch10_epoch500_all_lr0.005_640-r\model-230.pth",
                    type=str, help="path to model weights")
parser.add_argument("--data_path", type=str,
                    default=r"N:\AcademicResearchs\Datasets\IRSingleFrame_Dataset\SIRST-UAVB\SIRST-UAVB_OnlyUAV_Form",
                    help="dataset root directory (auto-detect format)")
parser.add_argument("--input_dir", default=None, type=str,
                    help="image directory (ignored when data_path is set)")
parser.add_argument("--output_dir", default=r'./eva_results', type=str, help="root directory used to store evaluation results")
parser.add_argument("--gt_json_path", type=str, default=None,
                    help="GT json path (auto-detected from data_path)")
parser.add_argument("--fixed_size", default=[640, 640], nargs='+', type=int, help="resize [H, W]")
parser.add_argument("--num_joints", default=1, type=int, help="number of keypoint joints")
parser.add_argument("--threshold", type=float, default=0.35, help="detection threshold")
parser.add_argument("--value_range", type=float, default=0.35, help="value range for NMS")
parser.add_argument("--max_num_targets", default=8, type=int, help="max number of targets")
parser.add_argument("--tp_distance", default=5, type=int, help="TP distance threshold")
parser.add_argument("--single_predict", default=False, type=str2bool, help="save single prediction images")
parser.add_argument("--contrast_predict", default=False, type=str2bool, help="save contrast images")
parser.add_argument("--save_heatmap", default=False, type=str2bool, help="save heatmap visualizations")
parser.add_argument("--save_json", default=True, type=str2bool, help="save COCO-style prediction JSON")
parser.add_argument("--json_eval", action="store_true",
                    help="skip model inference; run tools/eval_from_json on COCO JSON only")
parser.add_argument("--pred_json", type=str, default=None,
                    help="prediction JSON path; default: {output_dir}/predictions_coco.json")
parser.add_argument("--pdfa_thresholds", type=float, nargs="*", default=None,
                    help="confidence thresholds for PD/FA table (default: 0.1..1.0 step 0.1, see eval_from_json)")
parser.add_argument("--json_eval_report", type=str, default=None,
                    help="save full JSON-eval report to this path; default: {output_dir}/json_eval_report.txt")

global opt
opt = parser.parse_args()
opt.output_dir = resolve_output_subdir(
    opt.output_dir,
    infer_experiment_name_from_weights(opt.weights_path),
)


def _load_eval_from_json_module():
    """Load tools/eval_from_json.py (single source of truth)."""
    repo_root = os.path.abspath(os.path.dirname(__file__))
    mod_path = os.path.join(repo_root, "tools", "eval_from_json.py")
    if not os.path.isfile(mod_path):
        raise FileNotFoundError(
            "未找到 eval_from_json.py，请确认仓库中存在: " + mod_path
        )
    spec = importlib.util.spec_from_file_location("datasetsOperation_eval_from_json", mod_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["datasetsOperation_eval_from_json"] = mod
    spec.loader.exec_module(mod)
    return mod


def resolve_gt_json_path():
    """与 eval() 中一致的 GT 标注路径解析（供 json 复测）。"""
    if opt.gt_json_path:
        p = os.path.abspath(os.path.expanduser(opt.gt_json_path))
        return p if os.path.isfile(p) else None
    if not opt.data_path:
        return None
    fmt = detect_dataset_format(opt.data_path)
    if fmt == "format_b":
        p = os.path.join(opt.data_path, "annotations", "annotations.json")
    else:
        p = os.path.join(opt.data_path, "test", "annotations", "test.json")
    return p if os.path.isfile(p) else None


def run_json_eval_only():
    """对接 eval_from_json：对 SPIRE 导出的 predictions_coco.json 做指标与阈值表。"""
    efj = _load_eval_from_json_module()

    pred_path = opt.pred_json
    if not pred_path:
        pred_path = os.path.join(opt.output_dir, "predictions_coco.json")
    pred_path = os.path.abspath(os.path.expanduser(pred_path))
    if not os.path.isfile(pred_path):
        raise FileNotFoundError("预测 JSON 不存在: " + pred_path)

    gt_path = resolve_gt_json_path()
    if not gt_path:
        raise FileNotFoundError(
            "无法确定 GT JSON，请设置 --gt_json 或提供有效的 --data_path"
        )

    thresholds = opt.pdfa_thresholds
    if thresholds is not None and len(thresholds) == 0:
        thresholds = None

    tp_dist = float(opt.tp_distance)

    print("=" * 80)
    print("JSON-only evaluation (eval_from_json)")
    print("=" * 80)
    print(f"GT:  {gt_path}")
    print(f"Pred: {pred_path}")
    print(f"tp_distance: {tp_dist}")
    print(f"pdfa_thresholds: {thresholds or '[0.1..1.0 step 0.1]'}")
    print("-" * 80)

    results = efj.evaluate_from_json(
        gt_path,
        pred_path,
        tp_distance=tp_dist,
        pdfa_thresholds=thresholds,
    )

    m = results["metrics"]
    print("Overall metrics (eval_from_json; see datasetsOperation/eval_from_json.py for details):")
    print(f"  TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}")
    print(f"  Precision: {m['precision']:.6f}, Recall: {m['recall']:.6f}, F1: {m['f1_score']:.6f}")
    print(f"  Image count: {m['image_count']}")
    print("\nPD / FA (multi-threshold):")
    efj.print_pdfa_table(results["pdfa_table"])

    f1_list = results["pdfa_table"]["F1"]
    best_idx = f1_list.index(max(f1_list))
    best_t = results["pdfa_table"]["thresholds"][best_idx]
    print(f"\n★ Best F1 (on score threshold grid): {f1_list[best_idx]:.6f} @ thresh={best_t:.2f}")

    if opt.f:
        opt.f.write("\n--- json_eval (eval_from_json) ---\n")
        opt.f.write(f"GT: {gt_path}\nPred: {pred_path}\n")
        opt.f.write(f"TP: {m['tp']}, FP: {m['fp']}, FN: {m['fn']}\n")
        opt.f.write(f"Precision: {m['precision']:.6f}, Recall: {m['recall']:.6f}, F1: {m['f1_score']:.6f}\n")
        opt.f.write(f"Best F1 @ thresh: {f1_list[best_idx]:.6f} @ {best_t:.2f}\n")

    report_path = opt.json_eval_report
    if not report_path:
        report_path = os.path.join(opt.output_dir, "json_eval_report.txt")
    th_write = thresholds or [round(i * 0.1, 1) for i in range(1, 11)]
    efj.save_results(results, report_path, tp_dist, th_write)
    print(f"\nFull report saved to: {report_path}")
    if opt.f:
        opt.f.write(f"json_eval_report: {report_path}\n")

    bad = results["bad_image_ids"]
    if bad:
        print(f"\nError images ({len(bad)}): {', '.join(bad[:30])}{' ...' if len(bad) > 30 else ''}")
    print("=" * 80)


def process_image(img_path, gt_data, model, device, data_transform):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    original_img = Image.fromarray(img)

    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0).to(device)

    img_name = os.path.splitext(os.path.basename(img_path))[0]
    image_id = img_name

    with torch.no_grad():
        start = time.perf_counter()
        outputs = model(img_tensor)
        end = time.perf_counter()
        infer_time = end - start

    keypoints, scores = transforms.nrx_get_final_preds(
        outputs, [target["reverse_trans"]], True, opt.output_dir,
        img_name, opt.threshold, opt.value_range, opt.max_num_targets, opt.save_heatmap
    )
    keypoints = np.squeeze(keypoints)
    if keypoints.ndim == 1:
        keypoints = np.array([keypoints])
    scores = np.squeeze(scores)
    if scores.ndim == 0:
        scores = np.array([scores])
    scores = np.clip(scores, 0.0, 1.0)

    if gt_data:
        gt_keypoints = get_gt_keypoints(gt_data, image_id)
        gt_img = draw_keypoints(original_img.copy(), gt_keypoints, None,
                                r=3, draw_text=False, draw_scores=True)
    else:
        print("Did not find gt json files!!!")
        gt_img = original_img.copy()
        gt_keypoints = np.empty((0, 2))

    pred_img = draw_keypoints(original_img.copy(), keypoints, scores,
                              thresh=opt.threshold, r=3, draw_scores=True)

    if opt.contrast_predict:
        concatenated_img = np.concatenate(
            [np.array(original_img.copy()), np.array(gt_img), np.array(pred_img)], axis=1
        )
        concatenated_img_pil = Image.fromarray(concatenated_img)
        output_img_path = os.path.join(opt.output_dir, os.path.basename(img_path))
        concatenated_img_pil.save(output_img_path)

    width, height = original_img.size
    image_info = {"file_name": os.path.basename(img_path), "height": height, "width": width, "id": image_id}

    coco_kpts = []
    num_kpts = 0
    for i in range(len(keypoints)):
        x, y = keypoints[i]
        if x == 0 and y == 0:
            continue
        score = float(scores[i]) if i < len(scores) else 0.0
        coco_kpts.extend([float(x), float(y), score])
        num_kpts += 1

    annotation_info = {
        "id": None, "image_id": image_id, "keypoints": coco_kpts,
        "num_keypoints": num_kpts, "category_id": 1,
    }

    if opt.single_predict:
        single_dir = os.path.join(opt.output_dir, "single_predict")
        if not os.path.exists(single_dir):
            os.makedirs(single_dir)
        pred_img.save(os.path.join(single_dir, os.path.basename(img_path)))

    if gt_data is not None and len(gt_keypoints) > 0:
        tp, fp, fn = calculate_metrics(keypoints, gt_keypoints, opt.tp_distance)
    else:
        tp, fp, fn = 0, 0, 0

    bad_id = image_id if (fn > 0 or fp > 0) else None

    return tp, fp, fn, bad_id, image_info, annotation_info, infer_time


def eval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = SPIRENet(base_channel=32, num_joints=opt.num_joints)
    weights = torch.load(opt.weights_path, map_location=device)
    model.load_state_dict(weights if "model" not in weights else weights["model"])
    model.to(device)
    model.eval()

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1, 1), fixed_size=opt.fixed_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    input_dir = opt.input_dir
    gt_json_path = opt.gt_json_path
    img_idx_filter = None

    if opt.data_path:
        dataset_format = detect_dataset_format(opt.data_path)
        if dataset_format == 'format_b':
            input_dir = os.path.join(opt.data_path, "images")
            gt_json_path = os.path.join(opt.data_path, "annotations", "annotations.json")
            img_idx_file = os.path.join(opt.data_path, "img_idx", "test.txt")
            if os.path.exists(img_idx_file):
                with open(img_idx_file, 'r') as f:
                    img_idx_filter = set(line.strip() for line in f if line.strip())
                print(f"[INFO] format_b: images={input_dir}, test_count={len(img_idx_filter)}")
        elif dataset_format == 'format_a':
            input_dir = os.path.join(opt.data_path, "test", "test_images")
            gt_json_path = os.path.join(opt.data_path, "test", "annotations", "test.json")
            print(f"[INFO] format_a: images={input_dir}")

    if gt_json_path and os.path.exists(gt_json_path):
        with open(gt_json_path, 'r') as f:
            gt_data = json.load(f)
    else:
        gt_data = None

    total_tp = 0
    total_fp = 0
    total_fn = 0
    fn_fp_image_ids = []
    coco_images = []
    coco_annotations = []
    infer_times = []

    all_files = sorted(os.listdir(input_dir))
    if img_idx_filter is not None:
        all_files = [f for f in all_files if os.path.splitext(f)[0] in img_idx_filter]
        print(f"[INFO] Filtered test images: {len(all_files)}")

    for filename in all_files:
        img_path = os.path.join(input_dir, filename)
        if not os.path.isfile(img_path):
            continue

        tp, fp, fn, bad_id, image_info, annotation_info, infer_time = process_image(
            img_path, gt_data, model, device, data_transform
        )
        infer_times.append(infer_time)

        if gt_data:
            if bad_id is not None:
                fn_fp_image_ids.append(bad_id)
            total_tp += tp
            total_fp += fp
            total_fn += fn

        if opt.save_json:
            coco_images.append(image_info)
            coco_annotations.append(annotation_info)

    if infer_times:
        total_infer_time = sum(infer_times)
        avg_infer_time = total_infer_time / len(infer_times)
        print(f"Inference summary: {len(infer_times)} images, total {total_infer_time:.4f}s, avg {avg_infer_time:.6f}s/image")
        opt.f.write(
            f"Inference summary: {len(infer_times)} images, total {total_infer_time:.4f}s, avg {avg_infer_time:.6f}s/image\n"
        )

    if gt_data:
        precision, recall, f1_score = compute_prf1(total_tp, total_fp, total_fn)

        print(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}")

        opt.f.write(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}\n")
        opt.f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")

        fn_fp_image_ids_sorted = sorted(fn_fp_image_ids)
        with open(os.path.join(opt.output_dir, 'val_result.txt'), 'w') as f:
            f.write(f"TP: {total_tp:.4f}, FP: {total_fp:.4f}, FN: {total_fn:.4f}\n")
            f.write(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1_score:.4f}\n")
            f.write("FN, FP images id:\n")
            for image_id in fn_fp_image_ids_sorted:
                f.write(f'{image_id}\n')

    if opt.save_json:
        img_dict = {}
        for im in coco_images:
            img_dict[im["id"]] = im
        coco_images_unique = list(img_dict.values())

        for idx, ann in enumerate(coco_annotations, start=1):
            ann["id"] = idx

        pred_json = {
            "images": coco_images_unique,
            "annotations": coco_annotations,
            "categories": [{"id": 1, "name": "target"}]
        }

        pred_json_path = os.path.join(opt.output_dir, "predictions_coco.json")
        with open(pred_json_path, "w") as f:
            json.dump(pred_json, f, indent=2)
        print("COCO-style prediction JSON saved to:", pred_json_path)


if __name__ == '__main__':
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    opt.f = open(os.path.join(opt.output_dir, 'eval_' + time.ctime().replace(' ', '_').replace(':', '_') + '.txt'), 'w')
    print("SPIRE Evaluation")
    print(opt)
    opt.f.write(str(opt) + '\n')
    if opt.json_eval:
        try:
            run_json_eval_only()
        finally:
            opt.f.close()
        sys.exit(0)
    eval()
    opt.f.close()
