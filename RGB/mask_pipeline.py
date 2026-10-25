"""
mask_pipeline.py
The per-mask pose-hypothesis pipeline: DINO encoding, template matching,
2D-3D correspondence, RANSAC-PnP, Render-guided Re-Correspondence (RRC),
optional MegaPose refinement, and raw hypothesis scoring (cosine + IoU).
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from nearest_neighbor_utils import build_or_load_nn
from pnp import ransac_pnp
from visualization import (
    visualize_final_icp_refinement,
    visualize_high_level_nearest_neighbors,
    visualize_initial_pnp_poses,
    visualize_megapose_pose,
)


@dataclass
class MaskPipelineResult:
    cnos_score: float
    pnp_pose: np.ndarray
    rrc_pose: np.ndarray
    mega_pose: np.ndarray
    pnp_cos: float
    pnp_iou: float
    rrc_cos: float
    rrc_iou: float
    mega_cos: float
    mega_iou: float


class MaskPipeline:
    """Construct once per object, then call process_mask() per detected mask."""

    def __init__(
        self,
        configs,
        correspondence_finder,
        hypothesis_refiner,
        storage_full_path,
        obj_id,
        num_inplane_variations,
        inplane_rotation_angle,
        megapose_refiner=None,
        megapose_mode=None,
    ):
        self.configs = configs
        self.correspondence_finder = correspondence_finder
        self.hypothesis_refiner = hypothesis_refiner
        self.storage_full_path = storage_full_path
        self.obj_id = obj_id
        self.num_inplane_variations = num_inplane_variations
        self.inplane_rotation_angle = inplane_rotation_angle
        self.megapose_refiner = megapose_refiner
        self.megapose_mode = megapose_mode

    def process_mask(
        self,
        mask_idx,
        mask_raw,
        mask_score,
        raw_rgb,
        query_K_for_scale,
        query_K,
        high_level_query_rgb,
        high_level_query_mask,
        high_level_query_depth,
        nn_high_level,
    ) -> Optional[MaskPipelineResult]:
        """Returns a MaskPipelineResult, or None if the mask should be skipped
        (too few grid cells or too few correspondences)."""
        print(f"  Mask score: {mask_score:.4f}")

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

        # Single DINO pass: no depth sampling in RGB mode.
        query_embeddings, query_centers_depth, high_level_query_embedding = (
            self.correspondence_finder.encode_query_unified(
                query_rgb_for_dino, mask_raw
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

        if "visualize_high_level_nn" in self.configs.get("visualizations", []):
            visualize_high_level_nearest_neighbors(
                high_level_query_rgb,
                high_level_query_mask,
                top_views_closest_to_query,
                top_views_distances,
                input_to_high_level_encoder,
                self.configs["templates_dir"],
                self.obj_id,
                self.num_inplane_variations,
                self.inplane_rotation_angle,
                high_level_bg_weight=self.configs.get("high_level_bg_weight", 1.0),
            )

        (
            all_template_embeddings,
            all_template_indices,
            all_template_center_coor_values,
        ) = self.correspondence_finder.encode_nearest_templates(
            top_views_closest_to_query
        )

        if len(query_embeddings) <= 3:
            print(
                f"[MaskPipeline] Warning: Not enough DINOv2 grid cells "
                f"({len(query_embeddings)}) for mask index {mask_idx}. Skipping."
            )
            return None

        n_query = len(query_embeddings)
        query_patches_intrinsics = np.tile(
            query_K_for_scale[np.newaxis], (n_query, 1, 1)
        )
        query_patches_center_coors_and_value = query_centers_depth
        patches = None
        patches_depth = None

        k_patch = self.configs.get("num_patch_neighbors_to_search_for", 20)
        if self.configs.get("use_faiss_patch_nn", False):
            from nearest_neighbor_utils import faiss_kneighbors

            distances, nn_indices = faiss_kneighbors(
                all_template_embeddings,
                query_embeddings,
                k=k_patch,
                use_gpu=True,
            )
        else:
            nn = build_or_load_nn(
                all_template_embeddings,
                self.storage_full_path,
                k=k_patch,
                high_level=False,
            )
            distances, nn_indices = nn.kneighbors(query_embeddings)

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
                from sklearn.neighbors import NearestNeighbors as _NNRev

                nn_rev = _NNRev(n_neighbors=1, metric="cosine")
                nn_rev.fit(query_embeddings)
                _, rev_nn_indices = nn_rev.kneighbors(all_template_embeddings)
            top_k_tmpl = nn_indices[:, :K_use]
            rev_matches = rev_nn_indices.ravel()[top_k_tmpl]
            query_row_ids = np.arange(nn_indices.shape[0])[:, None]
            mnn_mask = rev_matches == query_row_ids
            n_before = top_k_tmpl.size
            n_after = mnn_mask.sum()
            print(
                f"[MNN] kept {n_after}/{n_before} mutual matches ({100*n_after/max(n_before,1):.1f}%)"
            )
        else:
            print(
                "[MNN] Mutual Nearest Neighbor filtering not enabled; skipping stats after MNN filtering."
            )

        query_patch_data_for_correspondence = (
            patches,
            patches_depth,
            query_patches_intrinsics,
            query_patches_center_coors_and_value,
            None,
        )
        template_data_for_correspondence = (
            all_template_indices,
            all_template_center_coor_values,
            all_template_embeddings,
        )

        corr_fn = (
            self.correspondence_finder.find_2d_3d_correspondences_vectorized
            if self.configs.get("use_vectorized_correspondences", False)
            else self.correspondence_finder.find_2d_3d_correspondences
        )
        (
            source_3d_points,
            target_2d_points,
            corr_weights,
            corr_template_embeddings,
            corr_raw_distances,
        ) = corr_fn(
            nn_indices,
            template_data_for_correspondence,
            query_patch_data_for_correspondence,
            num_neighbors=self.configs["num_patch_neighbors_to_use"],
            nn_distances=distances,
            mnn_mask=mnn_mask,
        )
        print(
            f"[Correspondences] Found {len(source_3d_points)} 2D-3D correspondences"
            f" from {len(query_embeddings)} query grid cells."
        )

        if len(source_3d_points) < 4:
            print(
                f"[MaskPipeline] Warning: Not enough 2D-3D correspondences "
                f"({len(source_3d_points)}) for PnP. Skipping."
            )
            return None
        print(
            f"[MaskPipeline] Estimating pose via RANSAC-PnP using "
            f"{len(source_3d_points)} correspondences..."
        )

        pnp_threshold = self.configs.get("pnp_reprojection_threshold", 8.0)
        print(f"[RANSAC-PnP] Using reprojection threshold of {pnp_threshold:.2f} px.")

        initial_ransac_estimated_poses = ransac_pnp(
            np.asarray(source_3d_points, dtype=np.float32),
            np.asarray(target_2d_points, dtype=np.float32),
            K=query_K,
            num_hypotheses=self.configs.get("num_ransac_hypotheses", 1),
            reprojection_threshold=pnp_threshold,
            max_iterations=self.configs.get("num_ransac_iterations", 1000),
            hypo_min_dist=self.configs["min_rot_dist_between_hypotheses"],
            weights=corr_weights,
            feature_distances=corr_raw_distances,
            soft_score_alpha=self.configs.get("soft_score_alpha", 0.7),
        )

        if not initial_ransac_estimated_poses:
            print(f"[MaskPipeline] No PnP pose found for mask {mask_idx}. Skipping.")
            return None
        initial_pnp_pose = initial_ransac_estimated_poses[0]

        # ── Render re-correspondence step (RRC) ───────────────────────────────
        rrc_pose = initial_pnp_pose
        if self.configs.get("render_re_correspondence", False):
            rrc_result = self.hypothesis_refiner.render_re_correspondence(
                initial_pose=initial_pnp_pose,
                query_K=query_K,
                query_embeddings=query_embeddings,
                query_centers_depth=query_centers_depth,
                img_h=high_level_query_rgb.shape[0],
                img_w=high_level_query_rgb.shape[1],
            )
            if rrc_result:
                rrc_pose = rrc_result[0]
                print("[RRC] Produced 1 refined pose.")
            else:
                print("[RRC] No poses returned; falling back to PnP pose.")

        # ── MegaPose refinement (input: rrc_pose only) ────────────────────────
        mega_min_corr = self.configs.get("megapose_min_correspondences", 15)
        final_pose = rrc_pose
        if self.megapose_refiner is not None:
            if len(source_3d_points) < mega_min_corr:
                print(
                    f"[MegaPose] Skipping: only {len(source_3d_points)} correspondences "
                    f"(minimum {mega_min_corr})."
                )
            else:
                print("[MegaPose] Refining RRC pose...")
                resolution = [
                    high_level_query_rgb.shape[0],
                    high_level_query_rgb.shape[1],
                ]
                camera_data = {"K": query_K, "resolution": resolution}
                depth_for_megapose = high_level_query_depth.copy() / 1000.0

                pose_for_megapose = rrc_pose.copy()
                pose_for_megapose[:3, 3] /= 1000.0

                try:
                    refined_result, _ = self.megapose_refiner.run_megapose_ref_batch(
                        self.hypothesis_refiner.obj_model_path,
                        rgb=high_level_query_rgb,
                        depth=(
                            depth_for_megapose
                            if self.megapose_mode == "rgbd"
                            else None
                        ),
                        camera_data=camera_data,
                        estimated_poses=np.array([pose_for_megapose]),
                        n_iterations=5,
                    )
                    refined_poses_np = (
                        refined_result.poses.detach().cpu().numpy().reshape(-1, 4, 4)
                    )
                    final_pose = refined_poses_np[0].copy()
                    final_pose[:3, 3] *= 1000.0
                    print("[MegaPose] Successfully refined 1 pose.")
                except Exception as e:
                    print(f"[MegaPose] Warning: Refinement failed: {e}")

        if "visualize_initial_pnp_poses" in self.configs.get("visualizations", []):
            visualize_initial_pnp_poses(
                initial_pnp_pose, rrc_pose, query_K, self.hypothesis_refiner
            )

        if "visualize_megapose_poses" in self.configs.get("visualizations", []):
            visualize_megapose_pose(final_pose, query_K, self.hypothesis_refiner)

        # ── Render-score all three pose variants (raw components) ────────────
        # We do NOT combine with CNOS score or apply weights yet. All per-mask
        # raw scores are collected by the caller and z-normalised together
        # after the mask loop, so the narrow absolute range of cosine
        # similarity (typically 0.80–0.99 even for bad poses) is expanded into
        # a meaningful relative signal before combining.
        raw_score_tuples = self.hypothesis_refiner.score_pose_batch(
            [initial_pnp_pose, rrc_pose, final_pose],
            high_level_query_rgb,
            high_level_query_mask,
            query_K,
        )
        pnp_cos, pnp_iou = raw_score_tuples[0]
        rrc_cos, rrc_iou = raw_score_tuples[1]
        mega_cos, mega_iou = raw_score_tuples[2]
        print(
            f"  [RawScore] mask={mask_idx} cnos={mask_score:.3f} | "
            f"pnp=(cos={pnp_cos:.3f},iou={pnp_iou:.3f}) "
            f"rrc=(cos={rrc_cos:.3f},iou={rrc_iou:.3f}) "
            f"mega=(cos={mega_cos:.3f},iou={mega_iou:.3f})"
        )

        if "visualize_final_icp" in self.configs.get("visualizations", []):
            visualize_final_icp_refinement(
                high_level_query_rgb,
                high_level_query_mask,
                query_K,
                [initial_pnp_pose, rrc_pose, final_pose],
                ["initial PnP", "RRC PnP", "MegaPose refined"],
                self.hypothesis_refiner,
            )

        return MaskPipelineResult(
            cnos_score=mask_score,
            pnp_pose=initial_pnp_pose,
            rrc_pose=rrc_pose,
            mega_pose=final_pose,
            pnp_cos=pnp_cos,
            pnp_iou=pnp_iou,
            rrc_cos=rrc_cos,
            rrc_iou=rrc_iou,
            mega_cos=mega_cos,
            mega_iou=mega_iou,
        )
