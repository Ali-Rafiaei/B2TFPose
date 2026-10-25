"""BOP dataset Dataset classes: BopQueryRawDINO (lightweight raw RGB/depth/
mask loader) and BopHighLevelCNOSMulti (multi-mask high-level query loader),
used by bop_inference.py. Works for any BOP dataset, not just YCBV."""

import json
import os
from typing import Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from segmentation import decode_masks, select_cnos_detections


class BopQueryRawDINO(Dataset):
    """
    Lightweight query dataset for the DINOv2-only pipeline.

    Identical to BopHighLevelCNOSMulti but skips center_point extraction
    entirely — DINOv2 uses the full foreground grid, not pre-sampled patch
    centers, so the erode-and-sample step is unnecessary overhead.

    __getitem__ returns None or a dict keyed by mask_id:
        {
            mask_id: {
                "mask":   np.ndarray (HxW uint8),
                "K":      np.ndarray (3x3),
                "img_id": str,
            },
            ...
            "__raw__": {
                "rgb":   np.ndarray (HxWx3 uint8),
                "depth": np.ndarray (HxW float32, depth_scale applied),
            }
        }
    """

    def __init__(
        self,
        dataset_root,
        object_id,
        mask_score_threshold: float = 0.4,
        cnos_num_masks: Union[int, str] = 2,
    ):
        self.dataset_root = dataset_root
        self.test_path = os.path.join(self.dataset_root, "test")
        self.obj_id = object_id
        test_list_path = os.path.join(self.dataset_root, "test_targets_bop19.json")
        with open(test_list_path, "r") as f:
            test_list = json.load(f)
        self.filtered_test_list = [
            item for item in test_list if item["obj_id"] == int(self.obj_id)
        ]

        segmentation_data_path = os.path.join(
            dataset_root, "cnos_segmentations/cnos_segs_reformatted.json"
        )
        with open(segmentation_data_path, "r") as f:
            segs = json.load(f)

        self.filtered_segs = [
            data for data in segs if data["category_id"] == int(self.obj_id)
        ]

        self.mask_score_threshold = mask_score_threshold
        self.cnos_num_masks = cnos_num_masks

    def __len__(self):
        return len(self.filtered_test_list)

    def __getitem__(self, idx):
        item_data = self.filtered_test_list[idx]
        scene_id = str(item_data["scene_id"]).zfill(6)
        img_id = str(item_data["im_id"]).zfill(6)

        scene_path = os.path.join(self.test_path, scene_id)

        with open(os.path.join(scene_path, "scene_camera.json"), "r") as f:
            scene_camera = json.load(f)
            camera_info = scene_camera[str(int(img_id))]
            depth_scale = camera_info["depth_scale"]
            K = np.array(camera_info["cam_K"]).reshape(3, 3)

        rgb_path = os.path.join(scene_path, "rgb", img_id + ".png")
        depth_path = os.path.join(scene_path, "depth", img_id + ".png")

        rgb = np.array(Image.open(rgb_path))
        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth * depth_scale

        # --- Mask selection (same logic as BopHighLevelCNOSMulti) ---
        all_obj_segmentations = [
            data
            for data in self.filtered_segs
            if data["scene_id"] == int(scene_id) and data["image_id"] == int(img_id)
        ]
        all_obj_segmentations = sorted(
            all_obj_segmentations, key=lambda x: x["score"], reverse=True
        )

        if len(all_obj_segmentations) == 0:
            return None

        selected = select_cnos_detections(
            all_obj_segmentations, self.mask_score_threshold, self.cnos_num_masks
        )
        masks, scores = decode_masks(selected)

        result = {"__raw__": {"rgb": rgb, "depth": depth}}

        for mask_id, mask in enumerate(masks):
            if mask.sum() == 0:
                result[mask_id] = None
                continue
            result[mask_id] = {
                "mask": mask,
                "K": K,
                "img_id": img_id,
                "mask_score": scores[mask_id],
            }

        if all(result.get(k) is None for k in result if k != "__raw__"):
            return None

        return result


class BopHighLevelCNOSMulti(Dataset):
    def __init__(
        self,
        dataset_root,
        object_id,
        mask_score_threshold: float = 0.4,
        cnos_num_masks: Union[int, str] = 2,
    ):
        self.dataset_root = dataset_root
        self.test_path = os.path.join(self.dataset_root, "test")
        self.obj_id = object_id
        test_list_path = os.path.join(self.dataset_root, "test_targets_bop19.json")

        segmentation_data_path = os.path.join(
            dataset_root, "cnos_segmentations/cnos_segs_reformatted.json"
        )
        with open(segmentation_data_path, "r") as f:
            segs = json.load(f)
        self.filtered_segs = [
            data for data in segs if data["category_id"] == int(self.obj_id)
        ]

        with open(test_list_path, "r") as f:
            test_list = json.load(f)
        self.filtered_test_list = [
            item for item in test_list if item["obj_id"] == int(self.obj_id)
        ]
        self.mask_score_threshold = mask_score_threshold
        self.cnos_num_masks = cnos_num_masks

    def __len__(self):
        return len(self.filtered_test_list)

    def __getitem__(self, idx):
        item_data = self.filtered_test_list[idx]
        scene_id = str(item_data["scene_id"]).zfill(6)
        img_id = str(item_data["im_id"]).zfill(6)

        scene_path = os.path.join(self.test_path, scene_id)

        with open(os.path.join(scene_path, "scene_camera.json"), "r") as f:
            scene_camera = json.load(f)
            camera_info = scene_camera[str(int(img_id))]
            depth_scale = camera_info["depth_scale"]

        rgb_path = os.path.join(scene_path, "rgb", img_id + ".png")
        depth_path = os.path.join(scene_path, "depth", img_id + ".png")

        rgb = np.array(Image.open(rgb_path))

        all_obj_segmentations = [
            data
            for data in self.filtered_segs
            if data["scene_id"] == int(scene_id) and data["image_id"] == int(img_id)
        ]
        all_obj_segmentations = sorted(
            all_obj_segmentations, key=lambda x: x["score"], reverse=True
        )

        if len(all_obj_segmentations) == 0:
            return None

        selected = select_cnos_detections(
            all_obj_segmentations, self.mask_score_threshold, self.cnos_num_masks
        )
        masks, _ = decode_masks(selected)

        depth = np.array(Image.open(depth_path), dtype=np.float32)
        depth = depth * depth_scale

        return rgb, masks, depth
