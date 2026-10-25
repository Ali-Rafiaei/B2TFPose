"""
refinement.py
Mesh rendering, Render-guided Re-Correspondence (RRC), and raw hypothesis
scoring (cosine similarity + IoU against the query crop). Shared by the
main pipeline's per-mask loop.

RGB has no depth, so there is no depth-consistency reweighting, no
depth/chamfer scoring, and no TeaserPP/ICP step here — RRC (re-render the
object at the current pose estimate and re-encode with DINO) is this
pipeline's own way of sharpening correspondences at test time, in place of
the depth-based pose-guided re-correspondence an RGBD pipeline would use.
"""

import cv2
import numpy as np
import pyrender
import torch
import trimesh
from PIL import Image
from torchvision import transforms as T

from image_patch_utils import crop_bbox_square_centered, extract_crop_with_coords
from pnp import compute_ransac_iterations, ransac_pnp


class HypothesisRefiner:
    """Construct once per object, reuse across every query image."""

    def __init__(
        self,
        configs: dict,
        obj_model_path: str,
        dataset_name: str,
        correspondence_finder,
    ):
        self.configs = configs
        self.obj_model_path = obj_model_path
        self.dataset_name = dataset_name
        self.correspondence_finder = correspondence_finder
        self.pyrender_mesh = None

        sz = correspondence_finder.dino_input_size
        self._render_preprocess = T.Compose(
            [
                T.Resize((sz, sz), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

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

    def render_pose_rgb(self, pose_4x4, K, width: int, height: int):
        results = self.render_poses_rgb_batch([pose_4x4], K, width, height)
        return results[0]

    def render_poses_rgb_batch(self, poses, K, width: int, height: int):
        """Batch render multiple poses by reusing a single scene."""
        mesh = self.load_textured_mesh()

        render_pose_adapter = np.eye(4, dtype=np.float32)
        render_pose_adapter[1, 1] = -1.0
        render_pose_adapter[2, 2] = -1.0

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
        if self.dataset_name == "ycbv":
            scene.add(
                pyrender.DirectionalLight(color=np.ones(3), intensity=50.5), pose=light_pose
            )
        else:
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
    # Render-guided Re-Correspondence (RRC)
    # =========================================================================

    def render_re_correspondence(
        self,
        initial_pose: np.ndarray,
        query_K: np.ndarray,
        query_embeddings: np.ndarray,
        query_centers_depth: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> list:
        """
        Render 10 axis-aligned rotation variants of the initial PnP pose, encode
        all renders with the same DINO model, pool embeddings into one combined
        set, match query patches against it to build 2D-3D correspondences, and
        run a single RANSAC-PnP to produce a refined pose estimate.

        Args:
            initial_pose:        (4,4) object-to-camera pose, translation in mm.
            query_K:             (3,3) camera intrinsic matrix for the query image.
            query_embeddings:    (N, D) L2-normalised DINO patch embeddings of
                                 the query image foreground patches.
            query_centers_depth: (N, 3) patch centres as (u, v, 0) in query image
                                 pixel coordinates (col-2 is a 0 sentinel; RGB mode).
            img_h, img_w:        Render canvas dimensions (pixels).

        Returns:
            List of (4,4) float64 pose matrices (object-to-camera, mm), or [] if
            insufficient correspondences were found.
        """
        from nearest_neighbor_utils import faiss_kneighbors

        # ── 1. Build 10 pose variants (initial + 9 axis-aligned rotations) ──
        _I = np.eye(3, dtype=np.float64)
        _rotation_deltas = [
            _I,  # original
            np.diag([1.0, -1.0, -1.0]),  # Rx180
            np.diag([-1.0, 1.0, -1.0]),  # Ry180
            np.diag([-1.0, -1.0, 1.0]),  # Rz180
            np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], dtype=np.float64),  # Rx90
            np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], dtype=np.float64),  # Rx270
            np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=np.float64),  # Ry90
            np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.float64),  # Ry270
            np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64),  # Rz90
            np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]], dtype=np.float64),  # Rz270
        ]

        all_poses = []
        R0 = initial_pose[:3, :3].astype(np.float64)
        t0 = initial_pose[:3, 3].astype(np.float64)
        for R_delta in _rotation_deltas:
            P = np.eye(4, dtype=np.float64)
            P[:3, :3] = R0 @ R_delta  # right-multiply = rotate in object frame
            P[:3, 3] = t0
            all_poses.append(P)

        # ── 2. Render all 10 poses ───────────────────────────────────────────
        render_results = self.render_poses_rgb_batch(all_poses, query_K, img_w, img_h)

        # ── 3. Encode all renders in one batched forward pass ───────────────
        #   Phase A: CPU crop extraction + preprocessing
        fx = query_K[0, 0]
        fy = query_K[1, 1]
        cx = query_K[0, 2]
        cy = query_K[1, 2]

        dino_model = self.correspondence_finder.dino_model

        padding_ratio = self.configs.get("template_crop_padding_ratio", 0.2)
        valid_tensors = []
        crop_meta = (
            []
        )  # (crop_x0, crop_y0, effective_side, mask_crop, render_depth, P_i)

        for P_i, (render_rgb, render_mask, render_depth) in zip(
            all_poses, render_results
        ):
            if render_mask.sum() == 0:
                continue
            ys, xs = np.where(render_mask > 0)
            max_dim = max(int(xs.max()) - int(xs.min()), int(ys.max()) - int(ys.min()))
            pad = int(max_dim * padding_ratio)
            effective_side = max(max_dim + 2 * pad, 32)
            center_x = (int(xs.min()) + int(xs.max())) / 2.0
            center_y = (int(ys.min()) + int(ys.max())) / 2.0
            rgb_crop, crop_x0, crop_y0 = extract_crop_with_coords(
                render_rgb, center_x, center_y, effective_side
            )
            mask_crop, _, _ = extract_crop_with_coords(
                render_mask, center_x, center_y, effective_side
            )
            valid_tensors.append(self._render_preprocess(Image.fromarray(rgb_crop)))
            crop_meta.append(
                (crop_x0, crop_y0, effective_side, mask_crop, render_depth, P_i)
            )

        if not valid_tensors:
            return []

        #   Phase B: single batched GPU forward pass
        use_intermediate = self.configs.get("dino_patch_use_intermediate_layer", False)
        n_blocks = len(dino_model.blocks)
        patch_layer_idx = (
            self.configs.get("dino_patch_intermediate_layer_idx", 18)
            if use_intermediate
            else n_blocks - 1
        )
        patch_apply_norm = self.configs.get("cache_v2_patch_apply_norm", False) or (
            not use_intermediate
        )

        inp = torch.stack(valid_tensors, dim=0).cuda()  # (N_valid, 3, sz, sz)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            outputs = dino_model.get_intermediate_layers(
                inp,
                n=[patch_layer_idx],
                norm=False,
                return_class_token=True,
            )
        patch_tokens = outputs[0][0].detach().float()  # (N_valid, P, D)
        if patch_apply_norm:
            patch_tokens = dino_model.norm(patch_tokens)

        feat_h = feat_w = int(patch_tokens.shape[1] ** 0.5)
        D = patch_tokens.shape[2]

        #   Phase C: CPU post-processing — reshape, mask, map coords, lift to 3D
        render_embeddings_list = []
        render_3d_list = []

        for b, (
            crop_x0,
            crop_y0,
            effective_side,
            mask_crop,
            render_depth,
            P_i,
        ) in enumerate(crop_meta):
            feat_2d = patch_tokens[b].reshape(feat_h, feat_w, D).detach().cpu().numpy()
            norms = np.linalg.norm(feat_2d, axis=-1, keepdims=True).clip(min=1e-6)
            feat_2d = feat_2d / norms  # L2-normalise

            mask_small = cv2.resize(
                mask_crop.astype(np.uint8),
                (feat_w, feat_h),
                interpolation=cv2.INTER_NEAREST,
            )
            rows, cols = np.where(mask_small > 0)
            if len(rows) == 0:
                continue

            emb = feat_2d[rows, cols]  # (M, D)
            u = (cols + 0.5) / feat_w * effective_side + crop_x0
            v = (rows + 0.5) / feat_h * effective_side + crop_y0

            ui = np.clip(np.round(u).astype(int), 0, img_w - 1)
            vi = np.clip(np.round(v).astype(int), 0, img_h - 1)
            d = render_depth[vi, ui].astype(np.float64)  # mm

            valid = d > 0
            if valid.sum() == 0:
                continue

            u_v = u[valid]
            v_v = v[valid]
            d_v = d[valid]
            X_cam = np.stack(
                [
                    (u_v - cx) * d_v / fx,
                    (v_v - cy) * d_v / fy,
                    d_v,
                ],
                axis=1,
            )  # (M, 3), mm, camera frame

            R_oc = P_i[:3, :3]
            t_oc = P_i[:3, 3]
            X_obj = (
                R_oc.T @ (X_cam.T - t_oc[:, np.newaxis])
            ).T  # (M, 3), mm, object frame

            render_embeddings_list.append(emb[valid].astype(np.float32))
            render_3d_list.append(X_obj.astype(np.float32))

        if not render_embeddings_list:
            return []

        # ── 4. Pool into one combined set ────────────────────────────────────
        all_emb = np.concatenate(render_embeddings_list, axis=0)  # (N_total, D)
        all_3d = np.concatenate(render_3d_list, axis=0)  # (N_total, 3)

        if len(all_emb) < 4:
            return []

        # ── 5. Match query patches against combined render embeddings ────────
        k_nn = self.configs.get("num_patch_neighbors_to_search_for", 3)
        k_nn = min(k_nn, len(all_emb))
        distances, nn_idx = faiss_kneighbors(
            all_emb, query_embeddings, k=k_nn, use_gpu=True
        )

        flat_nn = nn_idx.ravel()
        source_3d = all_3d[flat_nn]  # (N*k, 3)
        target_2d = np.repeat(query_centers_depth[:, :2], k_nn, axis=0)  # (N*k, 2)
        raw_w = distances.ravel().astype(np.float32)
        weights = np.exp(-raw_w)
        weights = weights / weights.sum()

        if len(source_3d) < 4:
            return []

        # ── 6. RANSAC-PnP on the new correspondences ─────────────────────────
        print(
            f"[RRC] Running RANSAC-PnP on {len(source_3d)} render correspondences "
            f"({len(render_embeddings_list)} renders contributed patches)."
        )
        return ransac_pnp(
            source_3d.astype(np.float32),
            target_2d.astype(np.float32),
            K=query_K,
            num_hypotheses=1,
            reprojection_threshold=self.configs.get("pnp_reprojection_threshold", 8.0),
            max_iterations=compute_ransac_iterations(len(source_3d)),
            hypo_min_dist=self.configs.get("min_rot_dist_between_hypotheses", 15.0),
            weights=weights,
            feature_distances=raw_w,
            soft_score_alpha=self.configs.get("soft_score_alpha", 0.7),
        )

    # =========================================================================
    # Hypothesis scoring
    # =========================================================================

    @staticmethod
    def zscore_normalize(arr: np.ndarray) -> np.ndarray:
        """
        Z-score normalise a 1-D score array across hypotheses.

        Maps every metric to mean=0, std=1 so weights directly control
        discriminative influence. Single-hypothesis case returns all-zeros.
        """
        std = float(arr.std())
        if std < 1e-9:
            return np.zeros_like(arr, dtype=np.float32)
        return ((arr - arr.mean()) / std).astype(np.float32)

    def score_pose_batch(
        self,
        poses: list,
        query_rgb: np.ndarray,
        query_mask: np.ndarray,
        query_K: np.ndarray,
    ) -> list:
        """
        Score N pose hypotheses against a single query crop/mask in one render pass.

        Returns a list of (cosine_sim, iou) tuples — one per input pose.
        Poses whose IoU falls below hypothesis_iou_threshold return (0.0, 0.0).

        Raw components are returned without any weighting or normalization so that
        the caller can z-normalize across all competing masks before combining.
        """
        H, W = query_rgb.shape[:2]
        batch_results = self.render_poses_rgb_batch(poses, query_K, W, H)

        iou_threshold = self.configs.get("hypothesis_iou_threshold", 0.2)

        valid_indices = []
        iou_values = []
        crops_render, crops_render_masks = [], []
        crops_query, crops_query_masks = [], []

        for i, (render_rgb, render_mask, _) in enumerate(batch_results):
            intersection = ((query_mask > 0) & (render_mask > 0)).sum()
            union = ((query_mask > 0) | (render_mask > 0)).sum()
            iou = float(intersection / (union + 1e-6))
            if iou < iou_threshold:
                continue
            valid_indices.append(i)
            iou_values.append(iou)

            intersection_mask = ((query_mask > 0) & (render_mask > 0)).astype(np.uint8)
            seg_render = render_rgb * render_mask[:, :, np.newaxis]
            seg_query = query_rgb * intersection_mask[:, :, np.newaxis]

            cr, crm = crop_bbox_square_centered(
                seg_render, query_mask, return_mask=True
            )
            cq, cqm = crop_bbox_square_centered(seg_query, query_mask, return_mask=True)
            crops_render.append(cr)
            crops_render_masks.append(crm)
            crops_query.append(cq)
            crops_query_masks.append(cqm)

        # Return sentinel (0, 0) for all poses when none pass the IoU gate
        if not valid_indices:
            return [(0.0, 0.0)] * len(poses)

        # Batch encode: [render_0, render_1, ..., query_0, query_1, ...]
        all_crops = crops_render + crops_query
        all_masks = crops_render_masks + crops_query_masks
        feats = self.correspondence_finder.encode_highlevel(all_crops, mask_np=all_masks)
        n_valid = len(valid_indices)
        render_feats = feats[:n_valid]
        query_feats = feats[n_valid:]

        scores_out = [(0.0, 0.0)] * len(poses)
        for j, i in enumerate(valid_indices):
            cos_sim = float(
                np.dot(render_feats[j], query_feats[j])
                / (
                    np.linalg.norm(render_feats[j]) * np.linalg.norm(query_feats[j])
                    + 1e-6
                )
            )
            scores_out[i] = (cos_sim, iou_values[j])

        return scores_out
