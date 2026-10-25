"""
refinement.py
Depth-consistency reweighting, pose-guided re-correspondence, mesh rendering,
and hypothesis scoring/selection. Shared by bop_inference.py and
live_demo_inference.py. Depth refinement itself (TeaserPP/ICP) lives in
teaserpp_refiner.py / icp_refinement_experimental.py, not here.
"""

from math import ceil
from typing import List

import numpy as np
import pyrender
import trimesh
from matplotlib import pyplot as plt
from scipy.spatial import cKDTree

from image_patch_utils import crop_bbox_square_centered
from geometry_3d_utils import rotation_error


class HypothesisRefiner:
    """Construct once per object, reuse across every query image."""

    def __init__(self, configs: dict, obj_model_path: str, obj_diameter: float, correspondence_finder):
        self.configs = configs
        self.obj_model_path = obj_model_path
        self.obj_diameter = obj_diameter
        self.correspondence_finder = correspondence_finder
        self.pyrender_mesh = None

    # =========================================================================
    # Rendering
    # =========================================================================

    def load_textured_mesh(self):
        """Loads and caches the mesh as a pyrender.Mesh (normalizes Scene/Trimesh)."""
        if self.pyrender_mesh is not None:
            return self.pyrender_mesh

        loaded = trimesh.load(self.obj_model_path, process=False, maintain_order=True)
        if isinstance(loaded, trimesh.Scene):
            geoms = list(loaded.geometry.values())
            if not geoms:
                raise ValueError(
                    f"[render_pose_rgb] ERROR: mesh at {self.obj_model_path} has no geometry."
                )
            loaded = trimesh.util.concatenate(geoms)
        if isinstance(loaded, trimesh.Trimesh):
            textured_render_mesh = loaded
        else:
            raise ValueError(
                f"[render_pose_rgb] ERROR: mesh at {self.obj_model_path} is not a Trimesh or Scene."
            )

        mesh_scale = float(self.configs.get("model_scale_to_mm", 1.0))
        if mesh_scale != 1.0:
            textured_render_mesh.apply_scale(mesh_scale)

        material = getattr(textured_render_mesh.visual, "material", None)
        if (
            textured_render_mesh.visual.kind != "texture"
            or getattr(material, "image", None) is None
        ):
            print(
                "[render_pose_rgb] WARNING: mesh has no UV texture; vertex colors will be used."
            )

        self.pyrender_mesh = pyrender.Mesh.from_trimesh(
            textured_render_mesh, smooth=False
        )

        return self.pyrender_mesh

    def render_pose_rgb(
        self, pose_4x4, K, width: int, height: int, query_rgb=None, query_mask=None
    ):
        """Renders the textured object at one pose; returns (rgb, mask)."""
        results = self.render_poses_rgb_batch([pose_4x4], K, width, height)
        return results[0]

    def render_poses_rgb_batch(self, poses, K, width: int, height: int):
        """Batch-renders poses by reusing one scene, updating only the mesh
        node pose between renders. Returns a list of (rgb, mask, depth)."""
        mesh = self.load_textured_mesh()

        render_pose_adapter = np.eye(4, dtype=np.float32)
        render_pose_adapter[1, 1] = -1.0
        render_pose_adapter[2, 2] = -1.0

        # Build scene once with a placeholder pose
        scene = pyrender.Scene(
            bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            ambient_light=np.array([0.03, 0.03, 0.03, 1.0], dtype=np.float32),
        )
        mesh_node = scene.add(mesh, pose=np.eye(4, dtype=np.float32))

        cam = pyrender.IntrinsicsCamera(
            fx=float(K[0, 0]),
            fy=float(K[1, 1]),
            cx=float(K[0, 2]),
            cy=float(K[1, 2]),
            znear=0.01,
            zfar=5000.0,
        )
        scene.add(cam, pose=np.eye(4, dtype=np.float32))

        light_pose = np.eye(4, dtype=np.float32)
        scene.add(
            pyrender.DirectionalLight(color=np.ones(3), intensity=5.5), pose=light_pose
        )
        scene.add(pyrender.PointLight(color=np.ones(3), intensity=5.0), pose=light_pose)

        renderer = pyrender.OffscreenRenderer(
            viewport_width=width, viewport_height=height
        )

        results = []
        for pose_4x4 in poses:
            pose_gl = (render_pose_adapter @ pose_4x4).astype(np.float32)
            scene.set_pose(mesh_node, pose_gl)

            color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            mask = (depth > 0).astype(np.uint8)
            rgb = color[..., :3]
            results.append((rgb, mask, depth))

        renderer.delete()
        return results

    # =========================================================================
    # Correspondence reweighting / re-correspondence
    # =========================================================================

    @staticmethod
    def reweight_by_depth_consistency(
        source_3d: list,
        target_3d: list,
        pose: np.ndarray,
        old_weights: np.ndarray,
        depth_threshold: float,
    ) -> np.ndarray:
        """Reweights correspondences by depth consistency: projects each
        source point through pose, compares predicted vs. measured depth
        (target_3d[i][2]), and downweights by exp(-error/depth_threshold).
        Returns old_weights unchanged if all correspondences are inconsistent."""
        # Both (N, 3).
        src = np.asarray(source_3d, dtype=np.float32)
        tgt = np.asarray(target_3d, dtype=np.float32)
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)

        # Vectorised projection: X_cam_pred = R @ X_obj + t, both (N, 3).
        X_cam_pred = src @ R.T + t
        d_pred = X_cam_pred[:, 2]
        d_meas = tgt[:, 2]

        depth_w = np.exp(-np.abs(d_pred - d_meas) / depth_threshold)
        new_weights = old_weights * depth_w
        total = new_weights.sum()
        if total < 1e-10:
            # All correspondences are inconsistent — fall back to old weights.
            return old_weights
        return (new_weights / total).astype(np.float32)

    @staticmethod
    def pose_guided_re_correspondence(
        source_3d: list,
        corr_template_embeddings: list,
        pose: np.ndarray,
        query_embeddings: np.ndarray,
        query_centers_depth: np.ndarray,
        query_K: np.ndarray,
        sim_threshold: float,
        max_spatial_dist: float,
    ) -> tuple:
        """Projects each template 3D point through pose into the query image,
        matches it to the nearest query grid-cell (by pixel distance + cosine
        similarity gate), and builds a new geometrically-constrained set of
        (source, target, weight) correspondences for a follow-up RANSAC-Horn
        call. Returns (new_src_3d, new_tgt_3d, new_weights); weights are empty
        if no correspondence passed both gates."""
        # (N, 3) and (N, D).
        src = np.asarray(source_3d, dtype=np.float32)
        tmpl_emb = np.asarray(corr_template_embeddings, dtype=np.float32)
        R = pose[:3, :3].astype(np.float32)
        t = pose[:3, 3].astype(np.float32)
        fx, fy = float(query_K[0, 0]), float(query_K[1, 1])
        cx, cy = float(query_K[0, 2]), float(query_K[1, 2])

        # Project all template 3D points into query image
        # (N, 3) camera frame
        X_cam = src @ R.T + t
        valid = X_cam[:, 2] > 0
        u_proj = np.where(valid, X_cam[:, 0] / X_cam[:, 2] * fx + cx, -1.0)
        v_proj = np.where(valid, X_cam[:, 1] / X_cam[:, 2] * fy + cy, -1.0)

        # Query grid-cell positions (u, v)
        # (N_q, 2)
        q_uv = query_centers_depth[:, :2]
        # (N, 2)
        proj_uv = np.stack([u_proj, v_proj], axis=1)

        # Spatial nearest-neighbor: projected point → query grid-cell
        tree = cKDTree(q_uv)
        # (N,), (N,)
        dists, nn_idx = tree.query(proj_uv, k=1)

        # Cosine similarity (both sides already L2-normalised)
        # (N, D)
        matched_q_emb = query_embeddings[nn_idx]
        # (N,) cosine sim
        sims = (tmpl_emb * matched_q_emb).sum(axis=1)

        # Filter
        # (N,) measured depths
        q_depths = query_centers_depth[nn_idx, 2]
        keep = valid & (dists < max_spatial_dist) & (sims >= sim_threshold) & (q_depths > 0)

        if not keep.any():
            return [], [], np.empty(0, dtype=np.float32)

        # Backproject accepted query grid-cells to camera-frame 3D
        keep_idx = np.where(keep)[0]
        new_src_3d = [source_3d[i] for i in keep_idx]
        new_tgt_3d = []
        for i in keep_idx:
            qi = nn_idx[i]
            u_q, v_q, d_q = query_centers_depth[qi]
            pt_x = (u_q - cx) * d_q / fx
            pt_y = (v_q - cy) * d_q / fy
            new_tgt_3d.append(np.array([pt_x, pt_y, d_q], dtype=np.float32))

        w = sims[keep_idx].astype(np.float32)
        w /= w.sum()

        return new_src_3d, new_tgt_3d, w

    # =========================================================================
    # Hypothesis scoring / selection
    # =========================================================================

    @staticmethod
    def zscore_normalize(arr: np.ndarray) -> np.ndarray:
        """Z-score normalizes a 1-D score array (mean=0, std=1); all-zeros if std≈0."""
        std = float(arr.std())
        if std < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr.mean()) / std).astype(np.float32)

    @staticmethod
    def depth_alignment_score(
        render_depth: np.ndarray,
        query_depth: np.ndarray,
        render_mask: np.ndarray,
        threshold: float,
    ) -> float:
        """Returns exp(-mean_depth_error/threshold) in (0, 1] over pixels where
        the render mask is active and both depths are valid; higher is better."""
        valid = (render_mask > 0) & (render_depth > 0) & (query_depth > 0)
        if valid.sum() < 10:
            return 0.0
        mean_err = float(np.mean(np.abs(render_depth[valid] - query_depth[valid])))
        return float(np.exp(-mean_err / threshold))

    def render_and_choose_best_pose(
        self,
        pose_estimations: dict,
        query_rgb: np.ndarray,
        query_masks: np.ndarray,
        query_K: np.ndarray,
        all_chamfer_dists: List[float],
        raw_depth: np.ndarray,
        visualize: bool = False,
    ):
        """Renders every (mask, hypothesis) candidate, scores all of them in one
        consistent pass, then reduces to one winner per mask, dedups (same
        physical instance detected under two masks), ranks, and truncates to
        num_object_instances (self.configs: int, or "all" when unknown, e.g.
        the live demo). Also reads dedup_translation_thresh_mm/
        dedup_rotation_thresh_deg from self.configs.

        Returns a list of (pose_4x4, mask_idx, pose_idx, hypo_idx), ranked
        best-first; empty if nothing passed the IoU gate.
        """

        H, W, _ = query_rgb.shape

        rendered_rgbs = []
        rendered_masks = []
        rendered_depths = []
        pose_list = []
        corr_mask_idx = []
        # per-mask hypothesis index for each valid pose in pose_list
        corr_hypo_idx = []
        cropped_rendered_rgbs = []
        cropped_query_rgbs = []
        cropped_rendered_masks = []
        cropped_query_masks = []
        chamfer_dists = []

        # Collect all poses and their mask associations for batch rendering
        all_poses_flat = []
        all_mask_keys = []
        # per-mask hypothesis index, used to look up pre-ICP poses
        all_hypo_idxs = []
        for k, v in pose_estimations.items():
            for hypo_idx, pose_4x4 in enumerate(v):
                all_poses_flat.append(pose_4x4)
                all_mask_keys.append(k)
                all_hypo_idxs.append(hypo_idx)

        # Batch render all poses at once (single scene setup)
        batch_results = self.render_poses_rgb_batch(all_poses_flat, query_K, W, H)

        counter = -1
        for idx, (render_rgb, render_mask, render_depth) in enumerate(batch_results):
            counter += 1
            k = all_mask_keys[idx]

            rendered_rgbs.append(render_rgb)
            rendered_masks.append(render_mask)

            # Require sufficient overlap between query and rendered masks.
            intersection = ((query_masks[k] > 0) & (render_mask > 0)).sum()
            union = ((query_masks[k] > 0) | (render_mask > 0)).sum()
            iou = intersection / (union + 1e-6)
            iou_threshold = self.configs.get("hypothesis_iou_threshold", 0.2)
            if iou < iou_threshold:
                # Skip this rendering as it does not overlap enough with the query mask
                continue

            # Focus the query on pixels where the model actually projects.
            # This removes background pixels that CNOS incorrectly included.
            intersection_mask = ((query_masks[k] > 0) & (render_mask > 0)).astype(
                np.uint8
            )
            segmented_rendered_rgb = render_rgb * render_mask[:, :, np.newaxis]
            segmented_query_rgb = query_rgb * intersection_mask[:, :, np.newaxis]
            cropped_rendered_rgb, cropped_render_mask = crop_bbox_square_centered(
                segmented_rendered_rgb, query_masks[k], return_mask=True
            )
            cropped_query_rgb, cropped_query_mask = crop_bbox_square_centered(
                segmented_query_rgb, query_masks[k], return_mask=True
            )
            cropped_rendered_rgbs.append(cropped_rendered_rgb)
            cropped_query_rgbs.append(cropped_query_rgb)
            cropped_rendered_masks.append(cropped_render_mask)
            cropped_query_masks.append(cropped_query_mask)
            corr_mask_idx.append(k)
            corr_hypo_idx.append(all_hypo_idxs[idx])
            pose_list.append(all_poses_flat[idx])
            chamfer_dists.append(all_chamfer_dists[counter])
            rendered_depths.append(render_depth)

        if len(pose_list) == 0:
            print("No valid pose hypotheses after IoU filtering.")
            return []
        # With this new approach, for each rendering we do one comparison with the
        #   corresponding cropped query rgb
        input_batch = cropped_rendered_rgbs + cropped_query_rgbs
        input_masks = cropped_rendered_masks + cropped_query_masks
        input_batch_feats = self.correspondence_finder.encode_highlevel(
            input_batch, mask_np=input_masks
        )
        middle_point = len(cropped_rendered_rgbs)
        query_feats = input_batch_feats[middle_point:]
        render_feats = input_batch_feats[:middle_point]
        scores = []
        for i in range(len(render_feats)):
            query_feat = query_feats[i]
            render_feat = render_feats[i]
            cos_sim = np.dot(render_feat, query_feat) / (
                np.linalg.norm(render_feat) * np.linalg.norm(query_feat) + 1e-6
            )
            # The higher the cosine similarity, the better the match
            scores.append(cos_sim)

        scores = np.array(scores)

        # Combined ranking: Cosine Similarity + Chamfer Distance
        # Cosine similarity scores (higher is better)
        cosine_sims = scores
        # Chamfer distances in mm
        chamfer_dists = np.array(chamfer_dists)

        # Adaptive threshold: 10% of object diameter, at least 10mm.
        chamfer_threshold = max(10.0, 0.1 * self.obj_diameter)
        print(
            f"\n[Adaptive Threshold] Object diameter: {self.obj_diameter:.1f}mm, Chamfer threshold: {chamfer_threshold:.1f}mm"
        )

        # Convert chamfer distances to scores using exponential decay
        # Lower chamfer = higher score
        chamfer_scores = np.exp(-chamfer_dists / chamfer_threshold)

        # Rendered depth vs. query depth score (appearance-gap-free geometric signal)
        depth_threshold = max(15.0, 0.08 * self.obj_diameter)
        depth_scores = np.array(
            [
                self.depth_alignment_score(
                    rendered_depths[i], raw_depth, rendered_masks[i], depth_threshold
                )
                for i in range(len(pose_list))
            ]
        )

        # Z-normalise each component so narrow absolute ranges (e.g. cosine 0.80-0.99)
        # become meaningful relative signals before combining with weights.
        weight_cosine = self.configs.get("score_weight_cosine", 0.25)
        weight_chamfer = self.configs.get("score_weight_chamfer", 0.35)
        weight_depth = self.configs.get("score_weight_depth", 0.40)

        cosine_z  = self.zscore_normalize(cosine_sims)
        chamfer_z = self.zscore_normalize(chamfer_scores)
        depth_z   = self.zscore_normalize(depth_scores)

        combined_scores = (
            weight_cosine * cosine_z
            + weight_chamfer * chamfer_z
            + weight_depth * depth_z
        )

        # ── Reduce to one winner per mask, dedup, rank, truncate ──────────────
        # Must stay a downstream step on the pool scored above, not a second
        # scoring pass, or per-mask winning scores stop being comparable.
        num_object_instances = self.configs.get("num_object_instances", 1)
        dedup_translation_thresh_mm = self.configs.get("dedup_translation_thresh_mm", 1e9)
        dedup_rotation_thresh_deg = self.configs.get("dedup_rotation_thresh_deg", 1e9)

        corr_mask_idx_arr = np.asarray(corr_mask_idx)
        # (score, flat_idx), one entry per distinct mask
        per_mask_winners = []
        for m in sorted(set(corr_mask_idx)):
            idxs = np.where(corr_mask_idx_arr == m)[0]
            best_local = idxs[np.argmax(combined_scores[idxs])]
            per_mask_winners.append((float(combined_scores[best_local]), int(best_local)))

        # Rank winners best-first
        per_mask_winners.sort(key=lambda pair: -pair[0])

        # Dedup: drop a winner if it's within both thresholds of an
        # already-kept winner (same physical instance, different mask/detection).
        kept = []
        for score, flat_idx in per_mask_winners:
            pose = pose_list[flat_idx]
            is_duplicate = False
            for _, kept_flat_idx in kept:
                kept_pose = pose_list[kept_flat_idx]
                trans_dist = float(np.linalg.norm(pose[:3, 3] - kept_pose[:3, 3]))
                rot_dist = rotation_error(pose, kept_pose)
                if trans_dist < dedup_translation_thresh_mm and rot_dist < dedup_rotation_thresh_deg:
                    is_duplicate = True
                    break
            if not is_duplicate:
                kept.append((score, flat_idx))
            if isinstance(num_object_instances, int) and len(kept) >= num_object_instances:
                break

        # top-ranked winner, used only for the visualization below
        best_idx = kept[0][1]

        if visualize:
            # Visualization to check performance
            plt.switch_backend("TkAgg")
            plt.figure(figsize=(30, 20))

            num_rows = 3
            num_cols = ceil((len(pose_list) + 2) / num_rows)
            plt.suptitle(
                f"Pose Hypotheses Evaluation via Rendering Comparison of bboxe crops, "
                f"Best is: {best_idx+1} ({len(kept)} instance(s) selected)",
                fontsize=24,
            )
            plt.subplot(num_rows, num_cols, 1)
            plt.imshow(query_rgb)
            plt.title("Query Patch")
            plt.axis("off")

            plt.subplot(num_rows, num_cols, 2)
            plt.imshow(cropped_query_rgbs[best_idx])
            plt.title("Cropped Query Patch")
            plt.axis("off")

            for i in range(len(pose_list)):
                plt.subplot(num_rows, num_cols, i + 3)
                plt.imshow(cropped_rendered_rgbs[i])
                plt.title(
                    f"Hypo{i+1} - Mask{corr_mask_idx[i]}\n"
                    f"Cos:{scores[i]:.4f} Chm:{chamfer_dists[i]:.1f} Dep:{depth_scores[i]:.4f}\n"
                    f"Combined:{combined_scores[i]:.4f}"
                )
                plt.axis("off")

            plt.show()
            plt.close()

        return [
            (pose_list[flat_idx], corr_mask_idx[flat_idx], flat_idx, corr_hypo_idx[flat_idx])
            for _, flat_idx in kept
        ]
