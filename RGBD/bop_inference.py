#   VARIANT: DINOv2 dense feature patch matching — RGBD pipeline
#   Uses depth to build 3D-3D correspondences, RANSAC-Horn solver,
#   ICP/TeaserPP for final alignment, and MegaPose RGBD refinement.
#   Template embedding cache (template_cache.py) builds patch H5 + global .npy
#   in one DINO forward call per batch, halving GPU work at startup.

import os
import sys

# Force EGL for offscreen rendering (works without a live X display, e.g. over SSH).
os.environ["PYOPENGL_PLATFORM"] = "egl"

# Repo root is two levels up; same layout for local and SLURM runs.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
# megapose/ is a git submodule (our fork of megapose6d); its package lives
# under src/, matching upstream's own layout.
sys.path.insert(0, os.path.join(REPO_ROOT, "megapose", "src"))

import json
import time as time_module

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
from io_utils import save_chosen_mask_idx, write_pose_to_csv_bop
from nearest_neighbor_utils import build_or_load_nn
from pipeline_setup import build_pipeline_components
from visualization import visualize_final_icp_refinement, visualize_pose_alignment_3d


class BopPoseEstimator:
    def __init__(self, configs):
        self.configs = configs

        self.inplane_rotation_angle = configs["inplane_rotation_angle"]
        self.num_inplane_variations = 360 // self.inplane_rotation_angle
        self.obj_id = configs["obj_id"]
        self.dataset_root = configs["dataset_root"]
        self.test_path = os.path.join(self.dataset_root, "test")
        self.dataset_name = os.path.basename(self.dataset_root.rstrip("/"))
        self.cnos_threshold = self.configs.get("cnos_score_threshold", 0.4)
        self.cnos_num_masks = self.configs.get("cnos_num_masks", 2)
        print(
            f"Dataset root: {self.dataset_root}, extracted dataset name: {self.dataset_name}"
        )

        self.obj_model_path = os.path.join(
            configs["dataset_root"], f"models/obj_{self.obj_id:06d}.ply"
        )

        # Object diameter for the RANSAC threshold
        models_info_path = os.path.join(
            configs["dataset_root"], "models/models_info.json"
        )
        with open(models_info_path, "r") as f:
            models_info = json.load(f)
        obj_diameter = models_info[str(self.obj_id)]["diameter"]

        test_list_path = os.path.join(
            configs["dataset_root"], "test_targets_bop19.json"
        )
        with open(test_list_path, "r") as f:
            test_list = json.load(f)
        self.filtered_test_list = [
            item for item in test_list if item["obj_id"] == int(self.obj_id)
        ]

        # BOP model vertices are already in mm — no scaling needed.
        components = build_pipeline_components(
            self.configs,
            obj_id=self.obj_id,
            obj_model_path=self.obj_model_path,
            dataset_name=self.dataset_name,
            obj_diameter=obj_diameter,
            model_scale_to_mm=1.0,
        )
        self.obj_model = components.obj_model
        self.storage_full_path = components.storage_full_path
        self.template_cache = components.template_cache
        self.hypothesis_refiner = components.hypothesis_refiner
        self.depth_refiner_type = components.depth_refiner_type
        self.teaserpp_refiner = components.teaserpp_refiner
        self.megapose_refiner = components.megapose_refiner
        self.mask_pipeline = components.mask_pipeline

    def estimate_pose(self):
        """Runs the full BOP test split; saves CSVs if log_estimates_csv is set."""

        # ── Query datasets & template cache ─────────────────────────────────
        # Lightweight loader: RGB/depth/mask only, no patch extraction yet.
        query_raw_dataset = BopQueryRawDINO(
            dataset_root=self.configs["dataset_root"],
            object_id=self.configs["obj_id"],
            mask_score_threshold=self.cnos_threshold,
            cnos_num_masks=self.cnos_num_masks,
            segmentation_model=self.configs.get("segmentation_model", "cnos"),
        )
        high_level_query_dataset = BopHighLevelCNOSMulti(
            dataset_root=self.configs["dataset_root"],
            object_id=self.configs["obj_id"],
            mask_score_threshold=self.cnos_threshold,
            cnos_num_masks=self.cnos_num_masks,
            segmentation_model=self.configs.get("segmentation_model", "cnos"),
        )

        self.template_cache_path = self.template_cache.ensure()

        high_level_template_embeddings = (
            self.template_cache.load_high_level_embeddings()
        )

        # Fit the nearest neighbors model on the template embeddings
        nn_high_level = build_or_load_nn(
            high_level_template_embeddings,
            self.storage_full_path,
            k=self.configs.get("num_high_level_nearest_templates_to_search_for", 20),
            high_level=True,
        )

        # Debug filters (unset by default): only_* limits to one image,
        # resume_* skips ahead to a (scene, img) pair and continues from there.
        debug_only_scene_id = self.configs.get("debug_only_scene_id")
        debug_only_img_id = self.configs.get("debug_only_img_id")
        debug_resume_scene_id = self.configs.get("debug_resume_scene_id")
        debug_resume_img_id = self.configs.get("debug_resume_img_id")
        resumed = debug_resume_scene_id is None and debug_resume_img_id is None

        # ── Per-image loop ───────────────────────────────────────────────────
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
            self.vis_file_prefix = f"sc{self.scene_id_int}_im{img_id}_obj{self.obj_id}"

            if (
                debug_only_scene_id is not None
                and self.scene_id_int != debug_only_scene_id
            ):
                continue
            if debug_only_img_id is not None and img_id != debug_only_img_id:
                continue
            if not resumed:
                if (
                    self.scene_id_int == debug_resume_scene_id
                    and img_id == debug_resume_img_id
                ):
                    resumed = True
                else:
                    continue

            print(
                f"Processing image: {img_id} of scene: {scene_id} for object {self.obj_id}..."
            )

            t_image_start = time_module.time()
            t0 = time_module.time()
            high_level_query_data = high_level_query_dataset[i]

            if high_level_query_data == None:
                continue

            high_level_query_rgb, high_level_query_masks, high_level_query_depth = (
                high_level_query_data
            )
            high_level_query_rgb = high_level_query_rgb[:, :, :3]

            # Load raw image data + center points (no patch extraction yet)
            query_raw_data = query_raw_dataset[i]
            if query_raw_data is None:
                print(f"Skipping data with index {i} due to no valid data.")
                continue

            raw_rgb = query_raw_data["__raw__"]["rgb"]
            raw_depth = query_raw_data["__raw__"]["depth"]

            all_horn_estimated_poses_all_masks = {}
            all_megapose_refined_poses_all_masks = {}
            all_icp_refined_poses_all_masks = {}
            all_chamfer_dists = []
            all_query_obj_points_cropped_all_masks = {}

            poses_to_save = []
            csv_labels = []

            t_data_load = time_module.time() - t0

            # Per-stage accumulators (summed over all mask hypotheses)
            t_acc_high_level_enc = 0.0
            t_acc_template_enc = 0.0
            t_acc_patch_match = 0.0
            t_acc_correspondence = 0.0
            t_acc_ransac_horn = 0.0
            t_acc_megapose = 0.0
            t_acc_depth_refiner = 0.0

            # Pre-load per-image JSON data (shared across all masks)
            with open(os.path.join(self.scene_path, "scene_camera.json"), "r") as f:
                scene_camera_all = json.load(f)

            # Sampled once per image, not per mask (sample_points_uniformly is
            # randomized and the mesh doesn't depend on which mask is active).
            model_points = np.asarray(
                self.obj_model.sample_points_uniformly(number_of_points=3000).points
            )

            # ── Per-mask loop: run the pose pipeline on each detected mask ──────
            for mask_idx, high_level_query_mask in enumerate(high_level_query_masks):
                print(
                    f"Processing mask index: {mask_idx} out of {len(high_level_query_masks)}"
                )
                if query_raw_data.get(mask_idx) is None:
                    print(
                        f"Skipping data with index {i} and mask index {mask_idx} due to no valid centers."
                    )
                    continue

                mask_raw = query_raw_data[mask_idx]["mask"]
                query_K_for_scale = query_raw_data[mask_idx]["K"]
                raw_img_id = query_raw_data[mask_idx]["img_id"]
                mask_score = query_raw_data[mask_idx]["mask_score"]

                # Official scene camera K (distinct from query_K_for_scale,
                # which is per-mask data from the raw dataset loader).
                scene_camera = scene_camera_all[str(int(img_id))]
                query_K = np.array(scene_camera["cam_K"]).reshape(3, 3)

                mask_result = self.mask_pipeline.process_mask(
                    mask_idx=mask_idx,
                    mask_raw=mask_raw,
                    mask_score=mask_score,
                    raw_rgb=raw_rgb,
                    raw_depth=raw_depth,
                    query_K_for_scale=query_K_for_scale,
                    query_K=query_K,
                    high_level_query_rgb=high_level_query_rgb,
                    high_level_query_mask=high_level_query_mask,
                    high_level_query_depth=high_level_query_depth,
                    nn_high_level=nn_high_level,
                    model_points=model_points,
                )
                if mask_result is None:
                    continue

                t_acc_high_level_enc += mask_result.timings["high_level_enc"]
                t_acc_template_enc += mask_result.timings["template_enc"]
                t_acc_patch_match += mask_result.timings["patch_match"]
                t_acc_correspondence += mask_result.timings["correspondence"]
                t_acc_ransac_horn += mask_result.timings["ransac_horn"]
                t_acc_megapose += mask_result.timings["megapose"]
                t_acc_depth_refiner += mask_result.timings["depth_refiner"]

                all_horn_estimated_poses_all_masks[mask_idx] = (
                    mask_result.horn_estimated_poses
                )
                all_megapose_refined_poses_all_masks[mask_idx] = (
                    mask_result.megapose_refined_poses
                )
                all_icp_refined_poses_all_masks[mask_idx] = (
                    mask_result.icp_refined_poses
                )
                all_chamfer_dists.extend(mask_result.chamfer_dists)
                all_query_obj_points_cropped_all_masks[mask_idx] = (
                    mask_result.query_obj_points_cropped
                )

            # ── Select the best hypothesis across masks ─────────────────────────
            if len(all_icp_refined_poses_all_masks) == 0:
                print("No estimated poses were found for any mask. Skipping...")
                continue
            t0 = time_module.time()
            if len(all_icp_refined_poses_all_masks) == 1:
                # Single mask — skip re-rendering/scoring, nothing to compare against.
                chosen_mask_idx = list(all_icp_refined_poses_all_masks.keys())[0]
                chosen_pose_idx = 0
                chosen_hypo_idx = 0
                selected_results = [
                    (
                        all_icp_refined_poses_all_masks[chosen_mask_idx][
                            chosen_pose_idx
                        ],
                        chosen_mask_idx,
                        chosen_pose_idx,
                        chosen_hypo_idx,
                    )
                ]
            else:
                # num_object_instances (default 1) is read from self.configs.
                selected_results = self.hypothesis_refiner.render_and_choose_best_pose(
                    all_icp_refined_poses_all_masks,
                    high_level_query_rgb,
                    high_level_query_masks,
                    query_K,
                    all_chamfer_dists,
                    raw_depth,
                    visualize="visualize_hypothesis_selection"
                    in self.configs.get("visualizations", []),
                )
            t_hypo_selection = time_module.time() - t0
            if not selected_results:
                print("No estimated pose was chosen. Skipping...")
                continue

            # ── Refine & save the selected pose(s) ───────────────────────────────
            # Usually runs once per image (num_object_instances defaults to 1).
            for (
                estimated_pose,
                chosen_mask_idx,
                chosen_pose_idx,
                chosen_hypo_idx,
            ) in selected_results:
                high_level_query_mask = high_level_query_masks[chosen_mask_idx]

                # Teaser++ deferred until after selection (only run on the winner).
                if self.depth_refiner_type == "teaserpp":
                    t0 = time_module.time()
                    depth_segmented_for_teaser = (
                        high_level_query_depth * high_level_query_mask
                    )
                    estimated_pose = self.teaserpp_refiner.refine_poses(
                        prediction=estimated_pose,
                        mask=high_level_query_mask.astype(np.uint8),
                        depth=depth_segmented_for_teaser,
                        K=query_K,
                    )
                    t_acc_depth_refiner += time_module.time() - t0

                # RANSAC-Horn pose for the chosen hypothesis (before ICP / MegaPose refinement)
                horn_pose = all_horn_estimated_poses_all_masks[chosen_mask_idx][
                    chosen_hypo_idx
                ]

                dataset_path = self.configs.get("dataset_root")
                dataset_name = os.path.basename(os.path.normpath(dataset_path))

                # Saving the fully refined pose
                poses_to_save.append(estimated_pose)
                depth_refiner_type = self.depth_refiner_type
                csv_labels.append(
                    (
                        "est-w-megapose-" + depth_refiner_type + "_" + dataset_name,
                        "-test",
                    )
                )

                # Saving the RANSAC-Horn pose for the chosen hypothesis
                poses_to_save.append(horn_pose)
                csv_labels.append(("est-horn-only" + "_" + dataset_name + "-test"))

                # Saving the MegaPose-only pose too, if MegaPose ran.
                megapose_refined_pose = all_megapose_refined_poses_all_masks[
                    chosen_mask_idx
                ][chosen_hypo_idx]
                poses_to_save.append(megapose_refined_pose)
                csv_labels.append(
                    ("est-w-megapose-only" + "_" + dataset_name + "-test")
                )

                print(
                    "Code reached the end of one mask iteration. Estimated pose appended to list for saving."
                )

                if "visualize_final_icp" in self.configs.get("visualizations", []):
                    visualize_final_icp_refinement(
                        high_level_query_rgb,
                        high_level_query_mask,
                        query_K,
                        poses_to_save,
                        csv_labels,
                        self.hypothesis_refiner,
                    )

                if "visualize_pose_alignment_3d" in self.configs.get(
                    "visualizations", []
                ):
                    visualize_pose_alignment_3d(
                        poses_to_save,
                        csv_labels,
                        model_points,
                        all_query_obj_points_cropped_all_masks[chosen_mask_idx],
                    )

            total_time = -1

            if self.configs["log_estimates_csv"]:
                self.save_pose_estimations(
                    poses_to_save,
                    csv_labels,
                    raw_img_id,
                    time=total_time,
                )
                save_chosen_mask_idx(
                    chosen_mask_idx,
                    img_id,
                    self.scene_id_int,
                    self.obj_id,
                    self.configs["results_dir"],
                )

    def save_pose_estimations(self, poses, labels, scene_id, scores=None, time=-1.0):
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
                time=time,
            )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Setup and load config
    parser = setup_argparser()
    args = parser.parse_args()

    # Determine config file
    if args.config:
        config_path = args.config
    else:
        default_config = os.path.join(
            os.path.dirname(__file__), "configs", "default_config_tudl.yaml"
        )
        if os.path.exists(default_config):
            config_path = default_config
            print(f"No config specified, using default: {default_config}")
        else:
            print("ERROR: No config file specified")
            import sys

            sys.exit(1)

    # Load config
    configs = load_config_with_overrides(config_path, args)
    print_config(configs, "Running YCBV Pose Estimation")
    save_run_metadata(configs, __file__)

    # Run pipeline
    import traceback

    exit_code = 0
    try:
        estimator = BopPoseEstimator(configs)
        estimator.estimate_pose()
    except BaseException:
        # Print and flush before os._exit bypasses all I/O cleanup.
        # On cluster nodes stderr is often redirected to a file; an explicit
        # flush ensures the traceback actually lands in the log.
        traceback.print_exc()
        sys.stderr.flush()
        sys.stdout.flush()
        exit_code = 1
    finally:
        # Explicitly stop the Panda3D renderer workers before exiting.
        # os._exit() does NOT kill child processes on Linux — they become orphans.
        # We must call stop() ourselves so workers exit their blocking queue.get().
        try:
            if (
                hasattr(estimator, "megapose_refiner")
                and estimator.megapose_refiner is not None
            ):
                estimator.megapose_refiner.close()
        except Exception:
            pass
        os._exit(exit_code)
