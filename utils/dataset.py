import os
import copy
import json

import torch
import numpy as np
import cv2
import torch.utils.data as data
from pycocotools.coco import COCO

from .metrics import detect_dataset_format


class IRST(data.Dataset):
    def __init__(self, root, dataset="train", transforms=None, det_json_path=None, fixed_size=(512, 512)):
        super().__init__()
        assert dataset in ["train", "test"], 'dataset must be in ["train", "test"]'
        assert os.path.exists(root), "file '{}' does not exist.".format(root)

        self.dataset_format = detect_dataset_format(root)

        if self.dataset_format == 'format_b':
            self.img_root = os.path.join(root, "images")
            self.anno_path = os.path.join(root, "annotations", "annotations.json")
            img_idx_file = os.path.join(root, "img_idx", f"{dataset}.txt")

            assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
            assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)
            assert os.path.exists(img_idx_file), "file '{}' does not exist.".format(img_idx_file)

            with open(img_idx_file, 'r') as f:
                self.split_img_ids = set(line.strip() for line in f if line.strip())
        else:
            anno_file = f"{dataset}.json"
            self.img_root = os.path.join(root, dataset, f"{dataset}_images")
            self.anno_path = os.path.join(root, dataset, "annotations", anno_file)
            self.split_img_ids = None

            assert os.path.exists(self.img_root), "path '{}' does not exist.".format(self.img_root)
            assert os.path.exists(self.anno_path), "file '{}' does not exist.".format(self.anno_path)

        self.fixed_size = fixed_size
        self.mode = dataset
        self.transforms = transforms
        self.coco = COCO(self.anno_path)

        with open(self.anno_path, 'r') as file:
            self.annotations = json.load(file)

        all_img_ids = list(sorted(self.coco.imgs.keys()))

        if self.split_img_ids is not None:
            img_ids = [img_id for img_id in all_img_ids if str(img_id) in self.split_img_ids]
        else:
            img_ids = all_img_ids

        self.target_list = []
        obj_idx = 0
        for img_id in img_ids:
            img_info = next((image for image in self.annotations["images"] if image["id"] == img_id), None)
            ann = next((anns for anns in self.annotations["annotations"] if anns["image_id"] == img_id), None)

            info = {
                "image_path": os.path.join(self.img_root, img_info["file_name"]),
                "image_id": img_id,
                "image_width": img_info['width'],
                "image_height": img_info['height'],
                "obj_index": obj_idx,
                "score": ann["score"] if ann is not None and "score" in ann else 1.
            }

            if det_json_path is None:
                keypoints = np.array(ann["keypoints"]).reshape([-1, 3])
                visible = keypoints[:, 2]
                keypoints = keypoints[:, :2]
                info["keypoints"] = keypoints
                info["visible"] = visible

            self.target_list.append(info)
            obj_idx += 1

    def __getitem__(self, idx):
        target = copy.deepcopy(self.target_list[idx])
        image = cv2.imread(target["image_path"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transforms is not None:
            image, person_info = self.transforms(image, target)
        return image, target

    def __len__(self):
        return len(self.target_list)

    @staticmethod
    def collate_fn(batch):
        imgs_tuple, targets_tuple = tuple(zip(*batch))
        imgs_tensor = torch.stack(imgs_tuple)
        return imgs_tensor, targets_tuple
