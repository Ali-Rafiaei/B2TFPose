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
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)
# megapose/ is a git submodule (our fork of megapose6d); its package lives
# under src/, matching upstream's own layout.
sys.path.insert(0, os.path.join(_REPO_ROOT, "megapose", "src"))

import json
import time as time_module

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from config_utils import (
    load_config_with_overrides,
    print_config,
    save_run_metadata,
    setup_argparser,
)
from nearest_neighbor_utils import build_or_load_nn
from pipeline_setup import build_pipeline_components
from segmentation import decode_masks, load_cnos_json, select_cnos_detections
from visualization import visualize_final_icp_refinement, visualize_pose_alignment_3d


class LiveDemoPoseEstimator:
    def __init__(self, configs):
        self.configs = configs

        self.inplane_rotation_angle = configs["inplane_rotation_angle"]
        self.num_inplane_variations = 360 // self.inplane_rotation_angle
        self.obj_id = configs["obj_id"]
        self.dataset_root = configs.get("dataset_root", "")
        self.cnos_threshold = self.configs.get("cnos_score_threshold", 0.4)
        self.cnos_num_masks = self.configs.get("cnos_num_masks", 2)

        # Config override, else the BOP-style layout (bop_inference.py always
        # uses the BOP-style path).
        if self.configs.get("model_path"):
            self.obj_model_path = self.configs["model_path"]
        else:
            self.obj_model_path = os.path.join(
                configs["dataset_root"], f"models/obj_{self.obj_id:06d}.ply"
            )

        # Config override, else None so build_pipeline_components estimates it
        # from the mesh (bop_inference.py always has a models_info.json lookup).
        obj_diameter = (
            float(self.configs["obj_diameter_mm"])
            if self.configs.get("obj_diameter_mm")
            else None
        )
        model_scale_to_mm = float(self.configs.get("model_scale_to_mm", 1.0))

        self.dataset_name = (
            self.configs.get("dataset_name")
            or (os.path.basename(self.configs["dataset_root"].rstrip("/"))
                if self.configs.get("dataset_root") else "custom")
        )
        print(f"Dataset name: {self.dataset_name}")

        components = build_pipeline_components(
            self.configs,
            obj_id=self.obj_id,
            obj_model_path=self.obj_model_path,
            dataset_name=self.dataset_name,
            obj_diameter=obj_diameter,
            model_scale_to_mm=model_scale_to_mm,
        )
        self.obj_model = components.obj_model
        self.storage_full_path = components.storage_full_path
        self.template_cache = components.template_cache
        self.hypothesis_refiner = components.hypothesis_refiner
        self.depth_refiner_type = components.depth_refiner_type
        self.teaserpp_refiner = components.teaserpp_refiner
        self.megapose_refiner = components.megapose_refiner
        self.mask_pipeline = components.mask_pipeline

    def run_demo(self, rgb_path, depth_path, cnos_json_path, K, depth_scale=1.0, output_dir="./demo_results"):
        """
        Single-image pose estimation for custom (non-BOP) data.

        Args:
            rgb_path:       Path to query RGB image (any format PIL can read).
            depth_path:     Path to query depth image (16-bit PNG typical).
            cnos_json_path: Path to CNOS detection JSON from inference_custom.py.
            K:              Camera intrinsics, np.ndarray shape (3, 3).
            depth_scale:    Multiply raw depth pixels by this to get mm (default 1.0).
            output_dir:     Directory where the result JSON is written.
        Returns:
            List of result dicts, one per successfully estimated mask, each containing
            obj_id, image, mask_idx, mask_score, R (3×3 list), t_mm (3-element list),
            time_s. Empty list if no mask yielded a valid pose.
        """
        raw_img_id = os.path.splitext(os.path.basename(rgb_path))[0]
        self.scene_id_int = 0
        self.vis_file_prefix = f"demo_{raw_img_id}_obj{self.obj_id}"

        # ── Load images ──────────────────────────────────────────────────────
        raw_rgb = np.array(Image.open(rgb_path).convert("RGB"))
        if depth_path.endswith(".npy"):
            raw_depth = np.load(depth_path).astype(np.float32) * depth_scale
        else:
            _depth_raw = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if _depth_raw is None:
                raise FileNotFoundError(f"Could not read depth image: {depth_path}")
            raw_depth = _depth_raw.astype(np.float32) * depth_scale

        # Note: Depth is in millimeters

        # ── Parse CNOS JSON → select → decode RLE masks ──────────────────────
        detections = load_cnos_json(cnos_json_path)
        if not detections:
            print("[run_demo] No valid detections found in CNOS JSON. Aborting.")
            return None

        selected_detections = select_cnos_detections(
            detections, self.cnos_threshold, self.cnos_num_masks
        )
        high_level_query_masks, mask_scores = decode_masks(selected_detections)

        high_level_query_rgb = raw_rgb
        high_level_query_depth = raw_depth

        print(f"[run_demo] Processing {len(high_level_query_masks)} mask(s).")

        # ── Template cache + KNN on global embeddings ─────────────────────────
        self.template_cache_path = self.template_cache.ensure()
        high_level_template_embeddings = self.template_cache.load_high_level_embeddings()
        nn_high_level = build_or_load_nn(
            high_level_template_embeddings,
            self.storage_full_path,
            k=self.configs.get("num_high_level_nearest_templates_to_search_for", 20),
            high_level=True,
        )

        t_image_start = time_module.time()
        t0 = time_module.time()

        # One result dict per selected instance, not one per mask processed.
        all_pose_results = []
        t_hypo_selection_total = 0.0

        # Per-stage timing accumulators (cumulative across all masks)
        t_acc_high_level_enc = 0.0
        t_acc_template_enc = 0.0
        t_acc_patch_match = 0.0
        t_acc_correspondence = 0.0
        t_acc_ransac_horn = 0.0
        t_acc_megapose = 0.0
        t_acc_depth_refiner = 0.0

        query_K = K.astype(np.float64)

        # Buffered across the per-mask loop below; selection runs once after.
        all_icp_refined_poses_all_masks = {}
        all_chamfer_dists = []
        all_query_obj_points_cropped_all_masks = {}

        # Sampled once per image, not per mask (sample_points_uniformly is
        # randomized and the mesh doesn't depend on which mask is active).
        model_to_mm = float(self.configs.get("model_scale_to_mm", 1.0))
        model_points = np.asarray(
            self.obj_model.sample_points_uniformly(number_of_points=3000).points
        ) * model_to_mm

        # ── Per-mask loop ─────────────────────────────────────────────────────
        for mask_idx, high_level_query_mask in enumerate(high_level_query_masks):
            print(f"Processing mask index: {mask_idx} out of {len(high_level_query_masks)}")

            mask_raw = high_level_query_mask
            query_K_for_scale = query_K
            mask_score = mask_scores[mask_idx]

            if mask_raw.sum() == 0:
                print(f"Skipping mask {mask_idx}: zero area.")
                continue

            correspondence_hook = None
            if "visualize_correspondences" in self.configs.get("visualizations", []):
                correspondence_hook = self._visualize_correspondence_units

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
                correspondence_debug_hook=correspondence_hook,
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

            # Buffer this mask's hypotheses; selection picks from the full pool.
            all_icp_refined_poses_all_masks[mask_idx] = mask_result.icp_refined_poses
            all_chamfer_dists.extend(mask_result.chamfer_dists)
            all_query_obj_points_cropped_all_masks[mask_idx] = mask_result.query_obj_points_cropped

        # ── Hypothesis selection across all masks ───────────────────────────────
        # num_object_instances ("all" in configs/demo_live.yaml) is read inside.
        if not all_icp_refined_poses_all_masks:
            print("[run_demo] No poses estimated for any mask.")
            return []

        t0 = time_module.time()
        selected_results = self.hypothesis_refiner.render_and_choose_best_pose(
            all_icp_refined_poses_all_masks,
            high_level_query_rgb,
            high_level_query_masks,
            query_K,
            all_chamfer_dists,
            raw_depth,
            visualize="visualize_hypothesis_selection" in self.configs.get("visualizations", []),
        )
        t_hypo_selection_total += time_module.time() - t0

        for estimated_pose, mask_idx, pose_idx, hypo_idx in selected_results:
            high_level_query_mask = high_level_query_masks[mask_idx]
            mask_score = mask_scores[mask_idx]
            query_obj_points_cropped = all_query_obj_points_cropped_all_masks[mask_idx]

            # ── TeaserPP deferred refinement ───────────────────────────────────
            if self.depth_refiner_type == "teaserpp":
                t0 = time_module.time()
                depth_segmented_for_teaser = high_level_query_depth * high_level_query_mask
                estimated_pose = self.teaserpp_refiner.refine_poses(
                    prediction=estimated_pose,
                    mask=high_level_query_mask.astype(np.uint8),
                    depth=depth_segmented_for_teaser,
                    K=query_K,
                )
                t_acc_depth_refiner += time_module.time() - t0

            # ── Visualizations ──────────────────────────────────────────────────
            if "visualize_final_icp" in self.configs.get("visualizations", []):
                visualize_final_icp_refinement(
                    high_level_query_rgb,
                    high_level_query_mask,
                    query_K,
                    [estimated_pose],
                    [f"mask_{mask_idx}"],
                    self.hypothesis_refiner,
                )
            if "visualize_pose_alignment_3d" in self.configs.get("visualizations", []):
                visualize_pose_alignment_3d(
                    [estimated_pose],
                    [f"mask_{mask_idx}"],
                    model_points,
                    query_obj_points_cropped,
                )

            # ── Collect result ──────────────────────────────────────────────────
            all_pose_results.append({
                "obj_id": int(self.obj_id),
                "image": rgb_path,
                "mask_idx": mask_idx,
                "mask_score": mask_score,
                "R": estimated_pose[:3, :3].tolist(),
                "t_mm": estimated_pose[:3, 3].tolist(),
            })
            print(f"[run_demo] Mask {mask_idx}: pose estimated — t = {estimated_pose[:3, 3]} mm")

        # ── Post-loop: timing + consolidated save ─────────────────────────────
        if not all_pose_results:
            print("[run_demo] No poses estimated for any mask.")
            return []

        total_time = (
            t_acc_high_level_enc + t_acc_template_enc
            + t_acc_patch_match + t_acc_correspondence + t_acc_ransac_horn
            + t_acc_megapose + t_acc_depth_refiner + t_hypo_selection_total
        )
        print(
            f"[Timing] total={total_time:.2f}s across {len(all_pose_results)} mask(s) | "
            f"hi_enc={t_acc_high_level_enc:.2f}s "
            f"tmpl_enc={t_acc_template_enc:.2f}s "
            f"match={t_acc_patch_match:.2f}s "
            f"corr={t_acc_correspondence:.2f}s "
            f"ransac={t_acc_ransac_horn:.2f}s "
            f"mega={t_acc_megapose:.2f}s "
            f"depth_ref={t_acc_depth_refiner:.2f}s "
            f"hypo_sel={t_hypo_selection_total:.2f}s"
        )

        os.makedirs(output_dir, exist_ok=True)
        per_mask_time = round(total_time / len(all_pose_results), 3)
        for r in all_pose_results:
            r["time_s"] = per_mask_time
        out_json = os.path.join(output_dir, f"{raw_img_id}_poses.json")
        with open(out_json, "w") as f:
            json.dump(all_pose_results, f, indent=2)
        print(f"\n[Result] {len(all_pose_results)} pose(s) written to {out_json}")
        for r in all_pose_results:
            print(f"  Mask {r['mask_idx']} (score {r['mask_score']:.3f}): t = {r['t_mm']} mm")

        return all_pose_results

    def _visualize_correspondence_units(
        self,
        source_3d_points,
        target_3d_points,
        query_rgb,
        query_K,
        mask_idx,
    ):
        """Unit-check viz for 3D-3D correspondences: a 2D reprojection figure
        (dots should land on the object) and a 3D figure (source dots should
        lie on the model surface)."""
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        # (N, 3) object frame, mm
        src = np.array(source_3d_points, dtype=np.float64)

        # (N, 3) camera frame, mm
        tgt = np.array(target_3d_points, dtype=np.float64)

        fx, fy = query_K[0, 0], query_K[1, 1]
        cx, cy = query_K[0, 2], query_K[1, 2]
        H, W = query_rgb.shape[:2]

        print(f"\n[Unit Check] Mask {mask_idx} — {len(src)} correspondences")
        print(f"  source (object frame) mm:"
              f"  X=[{src[:,0].min():.1f}, {src[:,0].max():.1f}]"
              f"  Y=[{src[:,1].min():.1f}, {src[:,1].max():.1f}]"
              f"  Z=[{src[:,2].min():.1f}, {src[:,2].max():.1f}]")
        print(f"  target (camera frame) mm:"
              f"  X=[{tgt[:,0].min():.1f}, {tgt[:,0].max():.1f}]"
              f"  Y=[{tgt[:,1].min():.1f}, {tgt[:,1].max():.1f}]"
              f"  Z=[{tgt[:,2].min():.1f}, {tgt[:,2].max():.1f}]")

        # ── 2D figure ────────────────────────────────────────────────────────
        tgt_z = tgt[:, 2]
        valid_tgt = tgt_z > 0
        tgt_u = (fx * tgt[valid_tgt, 0] / tgt_z[valid_tgt] + cx).astype(int)
        tgt_v = (fy * tgt[valid_tgt, 1] / tgt_z[valid_tgt] + cy).astype(int)
        in_bounds = (tgt_u >= 0) & (tgt_u < W) & (tgt_v >= 0) & (tgt_v < H)

        vis_tgt = query_rgb.copy()
        depths_inbounds = tgt_z[valid_tgt][in_bounds]
        if depths_inbounds.size:
            d_min, d_max = depths_inbounds.min(), depths_inbounds.max() + 1e-6
            cmap = plt.cm.jet
            for u, v, d in zip(tgt_u[in_bounds], tgt_v[in_bounds], depths_inbounds):
                c = cmap((d - d_min) / (d_max - d_min))
                color_bgr = (int(c[2]*255), int(c[1]*255), int(c[0]*255))
                cv2.circle(vis_tgt, (int(u), int(v)), 5, color_bgr, -1)

        plt.switch_backend("tkagg")
        fig2d, axes2d = plt.subplots(1, 2, figsize=(22, 10))
        axes2d[0].imshow(query_rgb)
        axes2d[0].set_title(f"Query RGB — mask {mask_idx}", fontsize=12)
        axes2d[0].axis("off")
        axes2d[1].imshow(vis_tgt)
        axes2d[1].set_title(
            f"Target (camera frame) re-projected  [{in_bounds.sum()}/{len(src)} in-frame]\n"
            f"Z: [{tgt[:,2].min():.0f}, {tgt[:,2].max():.0f}] mm   colour = depth",
            fontsize=11,
        )
        axes2d[1].axis("off")
        fig2d.suptitle(f"2D Correspondence Unit Check — mask {mask_idx}", fontsize=13)
        plt.tight_layout()
        plt.show()
        plt.close(fig2d)

        # ── 3D figure ────────────────────────────────────────────────────────
        # Sample object model for background cloud
        _scale = float(self.configs.get("model_scale_to_mm", 1.0))
        model_pts = np.asarray(self.obj_model.vertices, dtype=np.float64) * _scale
        if len(model_pts) > 4000:
            rng = np.random.default_rng(0)
            model_pts = model_pts[rng.choice(len(model_pts), 4000, replace=False)]

        fig3d = plt.figure(figsize=(22, 10))
        fig3d.suptitle(f"3D Correspondence Unit Check — mask {mask_idx}", fontsize=13)

        # Left: source correspondences on object model (object frame)
        ax_src = fig3d.add_subplot(1, 2, 1, projection="3d")
        ax_src.scatter(
            model_pts[:, 0], model_pts[:, 1], model_pts[:, 2],
            c="lightgrey", s=1, alpha=0.3, label="model surface",
        )
        ax_src.scatter(
            src[:, 0], src[:, 1], src[:, 2],
            c="red", s=12, alpha=0.8, label=f"source ({len(src)} pts)",
        )
        ax_src.set_xlabel("X (mm)"); ax_src.set_ylabel("Y (mm)"); ax_src.set_zlabel("Z (mm)")
        ax_src.set_title(
            f"Source correspondences on object model\n"
            f"X=[{src[:,0].min():.0f},{src[:,0].max():.0f}]  "
            f"Y=[{src[:,1].min():.0f},{src[:,1].max():.0f}]  "
            f"Z=[{src[:,2].min():.0f},{src[:,2].max():.0f}] mm",
            fontsize=10,
        )
        ax_src.legend(fontsize=9)
        ax_src.set_box_aspect([1, 1, 1])

        # Right: target correspondences in camera frame (3D)
        ax_tgt = fig3d.add_subplot(1, 2, 2, projection="3d")
        sc = ax_tgt.scatter(
            tgt[:, 0], tgt[:, 1], tgt[:, 2],
            c=tgt[:, 2], cmap="jet", s=12, alpha=0.8,
            label=f"target ({len(tgt)} pts)",
        )
        fig3d.colorbar(sc, ax=ax_tgt, shrink=0.5, label="Z (mm)")
        ax_tgt.set_xlabel("X (mm)"); ax_tgt.set_ylabel("Y (mm)"); ax_tgt.set_zlabel("Z (mm)")
        ax_tgt.set_title(
            f"Target correspondences in camera frame\n"
            f"X=[{tgt[:,0].min():.0f},{tgt[:,0].max():.0f}]  "
            f"Y=[{tgt[:,1].min():.0f},{tgt[:,1].max():.0f}]  "
            f"Z=[{tgt[:,2].min():.0f},{tgt[:,2].max():.0f}] mm",
            fontsize=10,
        )
        ax_tgt.legend(fontsize=9)
        ax_tgt.set_box_aspect([1, 1, 1])

        plt.tight_layout()
        plt.show()
        plt.close(fig3d)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def parse_intrinsics(s):
    """Parse camera intrinsics from a JSON file path or a 'fx,fy,cx,cy' string."""
    if os.path.isfile(s):
        with open(s) as f:
            d = json.load(f)
        return np.array(
            [[d["fx"], 0.0, d["cx"]], [0.0, d["fy"], d["cy"]], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    parts = s.split(",")
    if len(parts) != 4:
        raise ValueError(f"--intrinsics must be a JSON file or 'fx,fy,cx,cy', got: {s!r}")
    fx, fy, cx, cy = map(float, parts)
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


if __name__ == "__main__":
    import argparse as _argparse
    import random
    import traceback as _traceback
    import os as _os
    import sys as _sys

    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    default_config_path = "./configs/demo_offline_debug.yaml"
    default_rgb_path = "./live_demo_requests/20260609_141621_652313/frame.png"
    default_depth_path = "./live_demo_requests/20260609_141621_652313/frame.npy"
    default_cnos_json_path = "./cnos_debug_results/detections.json"

    parser = _argparse.ArgumentParser(description="B2TF RGBD Pose Estimation — custom data demo")
    parser.add_argument("--config",
                        default=default_config_path,
                        help="Path to demo YAML config (see configs/demo.yaml)")

    parser.add_argument("--rgb",
                        default=default_rgb_path,
                        help="Path to query RGB image")
    
    parser.add_argument("--depth",
                        default=default_depth_path,
                        help="Path to query depth image (e.g. 16-bit PNG)")

    parser.add_argument("--cnos_json",
                        default=default_cnos_json_path,
                        help="Path to CNOS detection JSON (output of inference_custom.py)")
    args = parser.parse_args()

    # All deployment settings (model path, intrinsics, etc.) come from the config file.
    configs = load_config_with_overrides(args.config, _argparse.Namespace())
    K = parse_intrinsics(configs["camera_intrinsics"])

    print_config(configs, "B2TF RGBD Demo")
    save_run_metadata(configs, __file__)

    _exit_code = 0
    estimator = None
    try:
        estimator = LiveDemoPoseEstimator(configs)
        estimator.run_demo(
            rgb_path=args.rgb,
            depth_path=args.depth,
            cnos_json_path=args.cnos_json,
            K=K,
            depth_scale=float(configs.get("depth_scale", 1.0)),
            output_dir=configs.get("output_dir", "./demo_results"),
        )
    except BaseException:
        _traceback.print_exc()
        _sys.stderr.flush()
        _sys.stdout.flush()
        _exit_code = 1
    finally:
        try:
            if (
                estimator is not None
                and hasattr(estimator, "megapose_refiner")
                and estimator.megapose_refiner is not None
            ):
                estimator.megapose_refiner.close()
        except Exception:
            pass
        _os._exit(_exit_code)
