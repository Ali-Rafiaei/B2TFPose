"""
mask_pipeline.py
The per-mask pose-hypothesis pipeline: DINO encoding, template matching,
3D-3D correspondence, RANSAC-Horn (+ depth-consistency-reweight + pose-guided
re-correspondence passes), optional MegaPose refinement, and per-hypothesis
ICP + Chamfer scoring.
"""

import time as time_module
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import open3d as o3d

from geometry_3d_utils import backproject_depth_to_3d
from horn import compute_ransac_iterations, ransac_horn
from icp_refinement_experimental import perform_icp
from nearest_neighbor_utils import build_or_load_nn
from visualization import (
    visualize_high_level_nearest_neighbors,
    visualize_pose_hypotheses_grid,
)


@dataclass
class MaskPipelineResult:
    icp_refined_poses: list
    chamfer_dists: list
    horn_estimated_poses: list
    megapose_refined_poses: list
    query_obj_points_cropped: np.ndarray
    timings: dict = field(default_factory=dict)


class MaskPipeline:
    """Construct once per object, then call process_mask() per detected mask."""

    def __init__(
        self,
        configs,
        correspondence_finder,
        hypothesis_refiner,
        storage_full_path,
        obj_model_path,
        ransac_threshold,
        depth_refiner_type,
        num_inplane_variations,
        inplane_rotation_angle,
        obj_id,
        megapose_refiner=None,
        megapose_mode=None,
    ):
        self.configs = configs
        self.correspondence_finder = correspondence_finder
        self.hypothesis_refiner = hypothesis_refiner
        self.storage_full_path = storage_full_path
        self.obj_model_path = obj_model_path
        self.ransac_threshold = ransac_threshold
        self.depth_refiner_type = depth_refiner_type
        self.num_inplane_variations = num_inplane_variations
        self.inplane_rotation_angle = inplane_rotation_angle
        self.obj_id = obj_id
        self.megapose_refiner = megapose_refiner
        self.megapose_mode = megapose_mode

    def process_mask(
        self,
        mask_idx,
        mask_raw,
        mask_score,
        raw_rgb,
        raw_depth,
        query_K_for_scale,
        query_K,
        high_level_query_rgb,
        high_level_query_mask,
        high_level_query_depth,
        nn_high_level,
        model_points,
        correspondence_debug_hook: Optional[Callable] = None,
    ) -> Optional[MaskPipelineResult]:
        """Returns a MaskPipelineResult, or None if the mask should be skipped
        (too few grid cells/correspondences, or no valid ICP hypothesis).
        correspondence_debug_hook, if given, runs right after correspondences
        are found — used by the live demo's debug view only."""
        timings = {
            "high_level_enc": 0.0,
            "template_enc": 0.0,
            "patch_match": 0.0,
            "correspondence": 0.0,
            "ransac_horn": 0.0,
            "megapose": 0.0,
            "depth_refiner": 0.0,
        }

        print(f"  Mask score: {mask_score:.4f}")

        # Depth centroid translation prior (for geometric template re-ranking)
        t_init_query = None
        if self.configs.get("use_geometric_template_reranking", False):
            t_init_query = self.correspondence_finder.compute_depth_centroid(
                raw_depth, mask_raw, query_K_for_scale
            )

        high_level_query_depth_segmented = (
            high_level_query_depth * high_level_query_mask
        )

        # Background attenuation for patch-level encoding.
        # patch_bg_weight=0.0 → hard black mask, 1.0 → original (no change).
        patch_bg_weight = self.configs.get("patch_bg_weight", 1.0)
        if patch_bg_weight < 1.0:
            query_rgb_for_dino = raw_rgb.astype(np.float32)
            query_rgb_for_dino[mask_raw == 0] *= patch_bg_weight
            query_rgb_for_dino = query_rgb_for_dino.clip(0, 255).astype(np.uint8)
        else:
            query_rgb_for_dino = raw_rgb
        input_to_high_level_encoder = query_rgb_for_dino

        t0 = time_module.time()
        # Single DINO pass: returns patch embeddings (with depth), global embedding
        query_embeddings, query_centers_depth, high_level_query_embedding = (
            self.correspondence_finder.encode_query_unified(
                query_rgb_for_dino,
                mask_raw,
                raw_depth,
                query_K_for_scale,
                0.0,
            )
        )
        high_level_query_embedding = high_level_query_embedding.reshape(1, -1)
        # D11 fix: n_neighbors passed explicitly rather than relying on nn_high_level's
        # own default, which may be stale if nn_index_high_level.joblib was cached by
        # an earlier run with a different num_high_level_nearest_templates_to_search_for.
        high_level_distances, high_level_indices = nn_high_level.kneighbors(
            high_level_query_embedding,
            n_neighbors=self.configs.get(
                "num_high_level_nearest_templates_to_search_for", 20
            ),
        )

        # Geometric IoU template re-ranking (optional)
        if (
            self.configs.get("use_geometric_template_reranking", False)
            and t_init_query is not None
        ):
            high_level_indices, high_level_distances = (
                self.correspondence_finder.rerank_templates_by_geo_iou(
                    high_level_indices,
                    high_level_distances,
                    mask_raw,
                    query_K_for_scale,
                    t_init_query,
                    raw_depth.shape,
                )
            )

        top_views_closest_to_query, top_views_distances = (
            self.correspondence_finder.find_high_level_neighbors_with_min_rot_dist(
                high_level_indices,
                high_level_distances,
                min_rot_dist_deg=self.configs.get(
                    "min_rot_dist_between_high_level_neighbors", 15
                ),
                num_neighbors_to_select=self.configs.get(
                    "num_high_level_nearest_templates_to_use", 20
                ),
            )
        )
        timings["high_level_enc"] += time_module.time() - t0

        if "visualize_high_level_nn" in self.configs.get("visualizations", []):
            visualize_high_level_nearest_neighbors(
                high_level_query_rgb,
                top_views_closest_to_query,
                top_views_distances,
                input_to_high_level_encoder,
                self.configs["templates_dir"],
                self.obj_id,
                self.num_inplane_variations,
                self.inplane_rotation_angle,
            )

        # Encode templates: read pre-computed DINOv2 patch embeddings from cache
        t0 = time_module.time()
        (
            all_template_embeddings,
            all_template_indices,
            all_template_center_coor_values,
            all_template_patches_rgbs,
        ) = self.correspondence_finder.encode_nearest_templates(
            top_views_closest_to_query
        )
        timings["template_enc"] += time_module.time() - t0

        if len(query_embeddings) <= 3:
            print(
                f"[MaskPipeline] Not enough DINOv2 grid cells ({len(query_embeddings)}) for mask {mask_idx}. Skipping."
            )
            return None

        # Build query data compatible with find_3d_2_3d_correspondences.
        n_query = len(query_embeddings)
        query_patches_intrinsics = np.tile(
            query_K_for_scale[np.newaxis], (n_query, 1, 1)
        )
        query_patches_center_coors_and_value = query_centers_depth
        patches = None
        patches_depth = None

        # Find the nearest neighbor for each query patch
        t0 = time_module.time()
        k_patch = self.configs.get("num_patch_neighbors_to_search_for", 20)
        if self.configs.get("use_faiss_patch_nn", False):
            from nearest_neighbor_utils import faiss_kneighbors

            distances, nn_indices = faiss_kneighbors(
                all_template_embeddings, query_embeddings, k=k_patch, use_gpu=True
            )
        else:
            nn = build_or_load_nn(
                all_template_embeddings,
                self.storage_full_path,
                k=k_patch,
                high_level=False,
            )
            distances, nn_indices = nn.kneighbors(query_embeddings)
        timings["patch_match"] += time_module.time() - t0

        # ── Mutual Nearest Neighbor (MNN) filtering ──────────────────────────
        mnn_mask = None
        if self.configs.get("use_mutual_nearest_neighbor", False):
            K_use = self.configs["num_patch_neighbors_to_use"]
            if self.configs.get("use_faiss_patch_nn", False):
                from nearest_neighbor_utils import faiss_kneighbors

                _, rev_nn_indices = faiss_kneighbors(
                    query_embeddings, all_template_embeddings, k=1, use_gpu=True
                )
            else:
                from sklearn.neighbors import NearestNeighbors

                nn_rev = NearestNeighbors(n_neighbors=1, metric="cosine")
                nn_rev.fit(query_embeddings)
                _, rev_nn_indices = nn_rev.kneighbors(all_template_embeddings)
            # mnn_mask[i, k] = True iff query i's match back-matches query i.
            top_k_tmpl = nn_indices[:, :K_use]
            rev_matches = rev_nn_indices.ravel()[top_k_tmpl]
            query_row_ids = np.arange(nn_indices.shape[0])[:, None]
            mnn_mask = rev_matches == query_row_ids
            n_before = top_k_tmpl.size
            n_after = mnn_mask.sum()
            print(
                f"[MNN] kept {n_after}/{n_before} mutual matches ({100*n_after/max(n_before,1):.1f}%)"
            )

        # query_indices not used in DINOv2 path (the trailing None below).
        query_patch_data_for_correspondence = (
            patches,
            patches_depth,
            query_patches_intrinsics,
            query_patches_center_coors_and_value,
            None,
        )
        # all_template_embeddings is needed for pose-guided re-correspondence.
        template_data_for_correspondence = (
            all_template_indices,
            all_template_center_coor_values,
            all_template_embeddings,
        )

        t0 = time_module.time()
        corr_fn = (
            self.correspondence_finder.find_3d_2_3d_correspondences_vectorized
            if self.configs.get("use_vectorized_correspondences", False)
            else self.correspondence_finder.find_3d_2_3d_correspondences
        )
        source_3d_points, target_3d_points, corr_weights, corr_template_embeddings = (
            corr_fn(
                nn_indices,
                template_data_for_correspondence,
                query_patch_data_for_correspondence,
                num_neighbors=self.configs["num_patch_neighbors_to_use"],
                nn_distances=distances,
                mnn_mask=mnn_mask,
            )
        )
        timings["correspondence"] += time_module.time() - t0
        print(
            f"[Correspondences] Found {len(source_3d_points)} 3D-3D correspondences"
            f" from {len(query_embeddings)} query grid cells."
        )

        if len(source_3d_points) < 3:
            print(
                f"[MaskPipeline] Not enough correspondences ({len(source_3d_points)}). Skipping mask {mask_idx}."
            )
            return None

        if correspondence_debug_hook is not None:
            correspondence_debug_hook(
                source_3d_points,
                target_3d_points,
                high_level_query_rgb,
                query_K,
                mask_idx,
            )

        print(f"[RANSAC-Horn] threshold = {self.ransac_threshold:.2f} mm")
        t0 = time_module.time()
        horn_estimated_poses = ransac_horn(
            source_3d_points,
            target_3d_points,
            threshold=self.ransac_threshold,
            max_iterations=compute_ransac_iterations(len(source_3d_points)),
            num_hypotheses=self.configs["num_ransac_hypotheses"],
            hypo_min_dist=self.configs["min_rot_dist_between_hypotheses"],
            early_stop_mode="ratio",
            weights=corr_weights,
        )
        timings["ransac_horn"] += time_module.time() - t0

        # ── Depth-consistency reweighting + second RANSAC-Horn pass ──────────
        if (
            self.configs.get("depth_consistency_reweighting", True)
            and len(horn_estimated_poses) > 0
        ):
            best_pose_for_reweight = horn_estimated_poses[0]
            depth_consistent_weights = (
                self.hypothesis_refiner.reweight_by_depth_consistency(
                    source_3d_points,
                    target_3d_points,
                    best_pose_for_reweight,
                    corr_weights,
                    depth_threshold=self.ransac_threshold,
                )
            )
            t0 = time_module.time()
            horn_estimated_poses = ransac_horn(
                source_3d_points,
                target_3d_points,
                threshold=self.ransac_threshold,
                max_iterations=compute_ransac_iterations(len(source_3d_points)),
                num_hypotheses=self.configs["num_ransac_hypotheses"],
                hypo_min_dist=self.configs["min_rot_dist_between_hypotheses"],
                early_stop_mode="ratio",
                weights=depth_consistent_weights,
            )
            timings["ransac_horn"] += time_module.time() - t0

        # ── Pose-guided re-correspondence + third RANSAC-Horn pass ───────────
        if (
            self.configs.get("pose_guided_re_correspondence", True)
            and len(horn_estimated_poses) > 0
        ):
            pgrc_src, pgrc_tgt, pgrc_weights = (
                self.hypothesis_refiner.pose_guided_re_correspondence(
                    source_3d_points,
                    corr_template_embeddings,
                    horn_estimated_poses[0],
                    query_embeddings,
                    query_centers_depth,
                    query_K,
                    sim_threshold=self.configs.get("pose_guided_sim_threshold", 0.5),
                    max_spatial_dist=self.configs.get(
                        "pose_guided_max_spatial_dist", 16.0
                    ),
                )
            )
            if len(pgrc_src) >= 3:
                print(
                    f"[PGRC] Re-correspondence: {len(pgrc_src)} geometrically-constrained matches."
                )
                t0 = time_module.time()
                horn_estimated_poses = ransac_horn(
                    pgrc_src,
                    pgrc_tgt,
                    threshold=self.ransac_threshold,
                    max_iterations=compute_ransac_iterations(len(pgrc_src)),
                    num_hypotheses=self.configs["num_ransac_hypotheses"],
                    hypo_min_dist=self.configs["min_rot_dist_between_hypotheses"],
                    early_stop_mode="ratio",
                    weights=pgrc_weights,
                )
                timings["ransac_horn"] += time_module.time() - t0
            else:
                print(
                    f"[PGRC] Too few re-correspondences ({len(pgrc_src)}), skipping third pass."
                )

        if "visualize_horn_poses" in self.configs.get("visualizations", []):
            _H_q, _W_q = high_level_query_rgb.shape[:2]
            visualize_pose_hypotheses_grid(
                horn_estimated_poses,
                query_K,
                width=_W_q,
                height=_H_q,
                hypothesis_refiner=self.hypothesis_refiner,
                title=f"Horn RANSAC Hypotheses — mask {mask_idx}",
            )

        # ── MegaPose refinement (per mask) ────────────────────────────────────
        # Default: no megapose.
        megapose_refined_poses = horn_estimated_poses
        mega_min_corr = self.configs.get("megapose_min_correspondences", 20)
        if self.megapose_refiner is not None:
            if len(source_3d_points) < mega_min_corr:
                print(
                    f"[MegaPose] Skipping mask {mask_idx}: only {len(source_3d_points)} "
                    f"correspondences (min {mega_min_corr})."
                )
            else:
                t0 = time_module.time()
                print(
                    f"[MegaPose] Refining {len(horn_estimated_poses)} RANSAC hypotheses for mask {mask_idx}..."
                )
                resolution = [
                    high_level_query_rgb.shape[0],
                    high_level_query_rgb.shape[1],
                ]
                camera_data = {"K": query_K, "resolution": resolution}
                # MegaPose expects meters, not millimeters.
                depth_for_megapose = high_level_query_depth.copy() / 1000.0
                poses_for_megapose = []
                for ep in horn_estimated_poses:
                    pm = ep.copy()
                    pm[:3, 3] /= 1000.0
                    poses_for_megapose.append(pm)
                try:
                    refined_result, _ = self.megapose_refiner.run_megapose_ref_batch(
                        self.obj_model_path,
                        rgb=high_level_query_rgb,
                        depth=(
                            depth_for_megapose if self.megapose_mode == "rgbd" else None
                        ),
                        camera_data=camera_data,
                        estimated_poses=np.array(poses_for_megapose),
                        n_iterations=5,
                    )
                    refined_poses_np = (
                        refined_result.poses.cpu().numpy().reshape(-1, 4, 4)
                    )
                    # Back to millimeters.
                    megapose_refined_poses = []
                    for rp in refined_poses_np:
                        rp[:3, 3] *= 1000.0
                        megapose_refined_poses.append(rp)
                    print(
                        f"[MegaPose] Successfully refined {len(megapose_refined_poses)} hypotheses"
                    )
                except Exception as e:
                    print(f"[MegaPose] Warning: Batch refinement failed: {e}")
                timings["megapose"] += time_module.time() - t0

        if "visualize_megapose_poses" in self.configs.get("visualizations", []):
            _H_q, _W_q = high_level_query_rgb.shape[:2]
            visualize_pose_hypotheses_grid(
                megapose_refined_poses,
                query_K,
                width=_W_q,
                height=_H_q,
                hypothesis_refiner=self.hypothesis_refiner,
                title=f"MegaPose-Refined Hypotheses — mask {mask_idx}",
            )

        use_megapose = self.megapose_refiner is not None

        # ── ICP / Chamfer per hypothesis ──────────────────────────────────────
        t0 = time_module.time()
        query_obj_points_cropped, _, _ = backproject_depth_to_3d(
            query_patches_intrinsics[0],
            high_level_query_depth_segmented,
        )
        pcd_query = o3d.geometry.PointCloud()
        pcd_query.points = o3d.utility.Vector3dVector(query_obj_points_cropped)

        icp_refined_poses = []
        chamfer_dists = []
        for hypo_pose in (
            megapose_refined_poses if use_megapose else horn_estimated_poses
        ):
            if self.depth_refiner_type == "teaserpp":
                # Teaser++ deferred until after selection — score as-is for now.
                final_est = hypo_pose
            else:
                _, final_est = perform_icp(
                    query_obj_points_cropped, model_points, hypo_pose
                )
            icp_refined_poses.append(final_est)
            # Chamfer distance on the refined pose, not pre-refinement.
            transferred_model_points = (
                final_est[:3, :3] @ model_points.T + final_est[:3, 3:4]
            ).T
            pcd_model = o3d.geometry.PointCloud()
            pcd_model.points = o3d.utility.Vector3dVector(transferred_model_points)
            dist_q2m = pcd_query.compute_point_cloud_distance(pcd_model)
            dist_m2q = pcd_model.compute_point_cloud_distance(pcd_query)
            chamfer_dist = (np.mean(dist_q2m) + np.mean(dist_m2q)) / 2.0
            chamfer_dists.append(chamfer_dist)
        timings["depth_refiner"] += time_module.time() - t0

        if not icp_refined_poses:
            print(f"[MaskPipeline] No valid ICP poses for mask {mask_idx}. Skipping.")
            return None

        return MaskPipelineResult(
            icp_refined_poses=icp_refined_poses,
            chamfer_dists=chamfer_dists,
            horn_estimated_poses=horn_estimated_poses,
            megapose_refined_poses=megapose_refined_poses,
            query_obj_points_cropped=query_obj_points_cropped,
            timings=timings,
        )
