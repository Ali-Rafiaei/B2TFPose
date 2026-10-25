#   VARIANT: DINO dense feature patch matching — RGB-only pipeline
#   No query depth: builds 2D-3D correspondences, RANSAC-PnP solver,
#   Render-guided Re-Correspondence (RRC), and MegaPose RGB refinement.
#   Template embedding cache (template_cache.py) builds patch H5 + global .npy
#   in one DINO forward call per batch, halving GPU work at startup.

import os
import sys

# Force EGL for all offscreen rendering. Pyglet requires a live X display
# which may not be available in SSH/tmux sessions even when $DISPLAY is set.
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Repo root is two levels up; local and SLURM runs share the same layout.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# megapose/ is a git submodule (our fork of megapose6d); its package lives
# under src/, matching upstream's own layout.
sys.path.insert(0, os.path.join(REPO_ROOT, "megapose", "src"))

import json

import numpy as np
import torch
from tqdm import tqdm

from config_utils import (
    load_config_with_overrides,
    print_config,
    save_run_metadata,
    setup_argparser,
)
from data_loaders import BopHighLevelCNOSMulti, BopQueryRawDINO
from io_utils import write_pose_to_csv_bop
from nearest_neighbor_utils import build_or_load_nn
from pipeline_setup import build_pipeline_components


class BopPoseEstimator:
    def __init__(self, configs):
        self.configs = configs

        self.obj_id = configs["obj_id"]
        self.dataset_root = configs["dataset_root"]
        self.test_path = os.path.join(self.dataset_root, "test")
        # Extracting the dataset name from the dataset root path keeping in mind that the
        # dataset root path can have a trailing slash or not, and that the dataset name is the last part of the path.
        self.dataset_name = os.path.basename(self.dataset_root.rstrip("/"))
        print(
            f"Dataset root: {self.dataset_root}, extracted dataset name: {self.dataset_name}"
        )
        self.cnos_threshold = self.configs.get("cnos_score_threshold", 0.4)
        self.cnos_num_masks = self.configs.get("cnos_num_masks", 2)

        self.obj_model_path = os.path.join(
            configs["dataset_root"], f"models/obj_{self.obj_id:06d}.ply"
        )

        test_list_path = os.path.join(
            configs["dataset_root"], "test_targets_bop19.json"
        )
        with open(test_list_path, "r") as f:
            test_list = json.load(f)
        self.filtered_test_list = [
            item for item in test_list if item["obj_id"] == int(self.obj_id)
        ]

        components = build_pipeline_components(
            self.configs,
            obj_id=self.obj_id,
            obj_model_path=self.obj_model_path,
            dataset_name=self.dataset_name,
        )
        self.storage_full_path = components.storage_full_path
        self.template_cache = components.template_cache
        self.correspondence_finder = components.correspondence_finder
        self.hypothesis_refiner = components.hypothesis_refiner
        self.mask_pipeline = components.mask_pipeline
        self.megapose_refiner = components.megapose_refiner

    def estimate_pose(self):
        query_raw_dataset = BopQueryRawDINO(
            dataset_root=self.configs["dataset_root"],
            object_id=self.configs["obj_id"],
            mask_score_threshold=self.cnos_threshold,
            cnos_num_masks=self.cnos_num_masks,
        )
        high_level_query_dataset = BopHighLevelCNOSMulti(
            dataset_root=self.configs["dataset_root"],
            object_id=self.configs["obj_id"],
            mask_score_threshold=self.cnos_threshold,
            cnos_num_masks=self.cnos_num_masks,
        )

        self.template_cache.ensure()
        high_level_template_embeddings = (
            self.template_cache.load_high_level_embeddings()
        )

        nn_high_level = build_or_load_nn(
            high_level_template_embeddings,
            self.storage_full_path,
            k=self.configs.get("num_high_level_nearest_templates_to_search_for", 20),
            high_level=True,
        )

        for i in tqdm(
            range(len(query_raw_dataset)),
            desc=f"Estimating Poses for Obj {self.obj_id}",
        ):
            item_data = self.filtered_test_list[i]
            scene_id = str(item_data["scene_id"]).zfill(6)
            self.scene_id_int = int(scene_id)
            img_id = str(item_data["im_id"]).zfill(6)

            self.scene_path = os.path.join(self.test_path, scene_id)
            img_id = int(img_id)

            print(
                f"Processing image: {img_id} of scene: {scene_id} for object {self.obj_id}..."
            )

            high_level_query_data = high_level_query_dataset[i]
            if high_level_query_data is None:
                continue

            high_level_query_rgb, high_level_query_masks, high_level_query_depth = (
                high_level_query_data
            )
            high_level_query_rgb = high_level_query_rgb[:, :, :3]

            query_raw_data = query_raw_dataset[i]
            if query_raw_data is None:
                print(f"Skipping data with index {i} due to no valid data.")
                continue

            raw_rgb = query_raw_data["__raw__"]["rgb"]

            poses_to_save = []
            csv_labels = []
            mask_scores_to_save = []
            # Raw per-mask scoring components — populated inside the mask loop and
            # z-normalised together after the loop so small absolute differences in
            # cosine similarity (which lives in 0.80–0.99) become meaningful.
            all_mask_entries = []  # list of MaskPipelineResult, one per successful mask

            with open(os.path.join(self.scene_path, "scene_camera.json"), "r") as f:
                scene_camera_all = json.load(f)

            for mask_idx, high_level_query_mask in enumerate(high_level_query_masks):
                if query_raw_data.get(mask_idx) is None:
                    print(
                        f"Skipping data with index {i} and mask index {mask_idx} due to no valid centers."
                    )
                    continue

                mask_raw = query_raw_data[mask_idx]["mask"]
                query_K_for_scale = query_raw_data[mask_idx]["K"]
                raw_img_id = query_raw_data[mask_idx]["img_id"]
                mask_score = query_raw_data[mask_idx]["mask_score"]

                print(
                    f"Processing mask: {mask_idx + 1} out of {len(high_level_query_masks)}\n"
                    f"  Mask score: {mask_score:.4f}\n"
                )

                scene_camera = scene_camera_all[str(int(img_id))]
                query_K = np.array(scene_camera["cam_K"]).reshape(3, 3)

                mask_result = self.mask_pipeline.process_mask(
                    mask_idx=mask_idx,
                    mask_raw=mask_raw,
                    mask_score=mask_score,
                    raw_rgb=raw_rgb,
                    query_K_for_scale=query_K_for_scale,
                    query_K=query_K,
                    high_level_query_rgb=high_level_query_rgb,
                    high_level_query_mask=high_level_query_mask,
                    high_level_query_depth=high_level_query_depth,
                    nn_high_level=nn_high_level,
                )
                if mask_result is None:
                    continue

                all_mask_entries.append(mask_result)

            # ── Post-mask-loop: z-normalise across all masks, then save ──────
            # All mask entries are collected first so that each scoring component
            # (cosine, IoU, CNOS) can be z-normalised across the N competing masks
            # before combining. This is critical because cosine similarity lives
            # in 0.80–0.99 — its absolute range is tiny, but the relative
            # differences across masks (e.g. 0.83 vs 0.91) are real signal.
            # Z-normalising expands those differences to a -2…+2 range that
            # contributes meaningfully to the combined score. CNOS score gets
            # the same treatment: a 0.45 vs 0.70 detection confidence difference
            # becomes a ±1 z-score rather than a small additive offset.
            if all_mask_entries and self.configs.get("log_estimates_csv", False):
                w_cos = self.configs.get("score_weight_cosine", 0.3)
                w_iou = self.configs.get("score_weight_iou", 0.7)
                w_cnos = self.configs.get("score_weight_cnos", 1.0)

                for cos_attr, iou_attr, pose_attr, variant, csv_label in [
                    ("pnp_cos", "pnp_iou", "pnp_pose", "pnp", f"pnp-only_{self.dataset_name}-test"),
                    ("rrc_cos", "rrc_iou", "rrc_pose", "rrc", f"pnp-rend-recorr_{self.dataset_name}-test"),
                    ("mega_cos", "mega_iou", "mega_pose", "mega", f"pnp-w-megapose_{self.dataset_name}-test"),
                ]:
                    cos_arr = np.array([getattr(e, cos_attr) for e in all_mask_entries])
                    iou_arr = np.array([getattr(e, iou_attr) for e in all_mask_entries])
                    cnos_arr = np.array([e.cnos_score for e in all_mask_entries])

                    cos_z = self.hypothesis_refiner.zscore_normalize(cos_arr)
                    iou_z = self.hypothesis_refiner.zscore_normalize(iou_arr)
                    cnos_z = self.hypothesis_refiner.zscore_normalize(cnos_arr)

                    combined = w_cos * cos_z + w_iou * iou_z + w_cnos * cnos_z

                    print(
                        f"  [{variant.upper()} z-scores] "
                        + "  ".join(
                            f"mask{j}:"
                            f"cos_z={cos_z[j]:+.2f} iou_z={iou_z[j]:+.2f} "
                            f"cnos_z={cnos_z[j]:+.2f} → {combined[j]:+.3f}"
                            for j in range(len(all_mask_entries))
                        )
                    )

                    for j, entry in enumerate(all_mask_entries):
                        poses_to_save.append(getattr(entry, pose_attr))
                        csv_labels.append(csv_label)
                        mask_scores_to_save.append(float(combined[j]))

                self.save_pose_estimations(
                    poses_to_save,
                    csv_labels,
                    raw_img_id,
                    scores=mask_scores_to_save,
                )

    def save_pose_estimations(self, poses, labels, scene_id, scores=None):
        object_id = self.configs["obj_id"]
        results_root = self.configs["results_dir"]
        for idx, (csv_label, pose_matrix) in enumerate(zip(labels, poses)):
            output_csv_path = os.path.join(results_root, f"{csv_label}.csv")
            pose_score = scores[idx] if scores is not None else 1.0
            write_pose_to_csv_bop(
                csv_path=output_csv_path,
                scene_id=self.scene_id_int,
                im_id=int(scene_id),
                obj_id=object_id,
                R=pose_matrix[:3, :3],
                t=pose_matrix[:3, 3:4],
                score=pose_score,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random
    import traceback as _traceback

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    parser = setup_argparser()
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        default_config = os.path.join(
            os.path.dirname(__file__), "configs", "default.yaml"
        )
        if os.path.exists(default_config):
            config_path = default_config
            print(f"No config specified, using default: {default_config}")
        else:
            print("ERROR: No config file specified.")
            sys.exit(1)

    configs = load_config_with_overrides(config_path, args)
    print_config(configs, "Running RGB Pose Estimation (PnP)")
    save_run_metadata(configs, __file__)

    _exit_code = 0
    estimator = None
    try:
        estimator = BopPoseEstimator(configs)
        estimator.estimate_pose()
    except BaseException:
        _traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        _exit_code = 1
    finally:
        try:
            if (
                estimator is not None
                and estimator.megapose_refiner is not None
            ):
                estimator.megapose_refiner.close()
        except Exception:
            pass
        import os as _os

        _os._exit(_exit_code)
