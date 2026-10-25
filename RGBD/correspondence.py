"""
correspondence.py
High-level template retrieval, patch-level DINO encoding, and 3D-3D
correspondence finding.
"""

from typing import List, Optional, Union

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T

from image_patch_utils import extract_crop_with_coords
from geometry_3d_utils import add_inplane_to_pose, rotation_error


class CorrespondenceFinder:
    """
    Construct once per object (after the DINO model is loaded and the template
    cache is set up), then use for every query image.
    """

    def __init__(
        self,
        configs: dict,
        dino_model,
        dino_preprocess,
        dino_input_size: int,
        num_inplane_variations: int,
        inplane_rotation_angle: int,
        list_of_template_transformations: np.ndarray,
        inv_template_transformations: np.ndarray,
        template_intrinsics: np.ndarray,
        geo_iou_model_pts: np.ndarray,
        template_cache,
    ):
        self.configs = configs
        self.dino_model = dino_model
        self.dino_preprocess = dino_preprocess
        self.dino_input_size = dino_input_size
        self.num_inplane_variations = num_inplane_variations
        self.inplane_rotation_angle = inplane_rotation_angle
        self.list_of_template_transformations = list_of_template_transformations
        self.inv_template_transformations = inv_template_transformations
        self.template_intrinsics = template_intrinsics
        self.geo_iou_model_pts = geo_iou_model_pts
        self.template_cache = template_cache

    # =========================================================================
    # High-level template retrieval
    # =========================================================================

    def find_high_level_neighbors_with_min_rot_dist(
        self,
        high_level_indices,
        high_level_distances,
        min_rot_dist_deg,
        num_neighbors_to_select,
    ):
        """
        Select high-level template neighbors ensuring a minimum
        rotational distance between them.

        params:
            high_level_indices: Indices of high-level nearest neighbors.
            high_level_distances: Distances of high-level nearest neighbors.
            min_rot_dist_deg: Minimum rotational distance in degrees.
            num_neighbors_to_select: Number of neighbors to select.
        """

        high_level_indices = high_level_indices[0]
        high_level_distances = high_level_distances[0]
        selected_rotations = []
        selected_neighbor_ids = []
        # Reconstruct each template's full pose, then rotation_error() gives
        # the angular distance between them.
        for idx, template_idx in enumerate(high_level_indices):
            base_template_idx = template_idx // self.num_inplane_variations
            inplane_rotation_deg = (
                template_idx % self.num_inplane_variations
            ) * self.inplane_rotation_angle
            base_template_pose = self.list_of_template_transformations[
                base_template_idx
            ]
            rotated_template_pose = add_inplane_to_pose(
                base_template_pose, inplane_rotation_deg
            )
            # Check rotational distance with already selected neighbors
            is_valid = True
            for sel_rot in selected_rotations:
                rot_dist = rotation_error(
                    rotated_template_pose[:3, :3], sel_rot[:3, :3]
                )
                if rot_dist < min_rot_dist_deg:
                    is_valid = False
                    break
            if is_valid:
                selected_neighbor_ids.append(idx)
                selected_rotations.append(rotated_template_pose)
            if len(selected_neighbor_ids) >= num_neighbors_to_select:
                break

        selected_neighbors = high_level_indices[selected_neighbor_ids]
        selected_distances = high_level_distances[selected_neighbor_ids]

        return selected_neighbors, selected_distances

    def rerank_templates_by_geo_iou(
        self,
        hl_indices: np.ndarray,
        hl_distances: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
        t_init: np.ndarray,
        query_img_shape: tuple,
    ) -> tuple:
        """
        Re-rank KNN template candidates by fusing appearance score with geometric IoU.

        Projects the CAD model using each template's rotation and the depth-centroid
        translation t_init, computes IoU with the query mask, then blends with the
        appearance (cosine) score using the SAM-6D formula:
            fused = (appearance + alpha * geo_iou * visibility) / (1 + visibility)

        Args:
            hl_indices:      (1, K) int    — raw KNN template indices.
            hl_distances:    (1, K) float  — cosine distances in [0, 2].
            mask:            (H, W) uint8  — binary query mask.
            K:               (3, 3) float  — query camera intrinsics.
            t_init:          (3,) float32  — depth centroid in mm (camera frame).
            query_img_shape: (H, W, ...)   — raw image shape for silhouette bounds.

        Returns:
            (1, K) int, (1, K) float — re-sorted indices and distances.
        """
        alpha = self.configs.get("geo_reranking_alpha", 0.5)
        proj_radius = self.configs.get("geo_reranking_proj_radius", 3)

        n = hl_indices.shape[1]
        # sklearn cosine distance ∈ [0, 2]; convert to similarity ∈ [-1, 1]
        # (K,)
        appearance_scores = 1.0 - hl_distances[0]

        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        H, W = query_img_shape[:2]
        # (N, 3) mm
        model_pts = self.geo_iou_model_pts

        geo_iou_scores = np.zeros(n, dtype=np.float32)
        visibility_scores = np.zeros(n, dtype=np.float32)

        for j, tmpl_idx in enumerate(hl_indices[0]):
            base_id = int(tmpl_idx) // self.num_inplane_variations
            # (3, 3)
            R = self.list_of_template_transformations[base_id, :3, :3]

            # Project: X_cam = R @ X_obj + t_init
            # (N, 3)
            X_cam = model_pts @ R.T + t_init[np.newaxis]
            valid_z = X_cam[:, 2] > 0
            if valid_z.sum() < 5:
                continue

            X_v = X_cam[valid_z]
            u = (fx * X_v[:, 0] / X_v[:, 2] + cx).astype(np.int32)
            v = (fy * X_v[:, 1] / X_v[:, 2] + cy).astype(np.int32)

            in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
            vis = in_bounds.sum() / max(valid_z.sum(), 1)
            visibility_scores[j] = float(vis)

            if in_bounds.sum() < 5:
                continue

            u_b = u[in_bounds]
            v_b = v[in_bounds]

            # Build projected silhouette via point dilation
            proj_mask = np.zeros((H, W), dtype=np.uint8)
            proj_mask[v_b.clip(0, H - 1), u_b.clip(0, W - 1)] = 1
            if proj_radius > 0:
                k = 2 * proj_radius + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
                proj_mask = cv2.dilate(proj_mask, kernel)

            intersection = int(np.logical_and(proj_mask > 0, mask > 0).sum())
            union = int(np.logical_or(proj_mask > 0, mask > 0).sum())
            if union > 0:
                geo_iou_scores[j] = intersection / union

        # SAM-6D fused score
        fused = (appearance_scores + alpha * geo_iou_scores * visibility_scores) / (
            1.0 + visibility_scores.clip(min=1e-6)
        )

        sort_order = np.argsort(-fused)
        if self.configs.get("use_geometric_template_reranking", False):
            top5_geo = geo_iou_scores[sort_order[:5]]
            print(f"[GeoIoU] top-5 geo_iou after reranking: {np.round(top5_geo, 3)}")

        return hl_indices[:, sort_order], hl_distances[:, sort_order]

    def encode_nearest_templates(self, top_nearest_templates: np.ndarray):
        """
        Return pre-computed DINO patch embeddings for the requested template views.

        Opens the H5 cache built by TemplateCache.build_caches() and reads only the
        requested tmpl_idx datasets — O(K) dict lookups, zero GPU work.

        Returns the same 4-tuple as the original encoding methods.
        """
        parts_emb, parts_idx, parts_ctr = [], [], []

        with h5py.File(self.template_cache.resolved_path, "r") as f:
            tmpls = f["templates"]
            for tmpl_idx in top_nearest_templates:
                key = str(int(tmpl_idx))
                if key not in tmpls:
                    continue
                # (K_i, D)
                emb = tmpls[key]["embeddings"][:]
                # (K_i, 3)
                ctr = tmpls[key]["centers"][:]
                parts_emb.append(emb)
                parts_idx.append(np.full(len(emb), tmpl_idx, dtype=np.int64))
                parts_ctr.append(ctr)

        if not parts_emb:
            D = self.dino_model.embed_dim
            return (
                np.zeros((0, D)),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3)),
                np.array([]),
            )

        all_embeddings = np.concatenate(parts_emb, axis=0)
        all_indices = np.concatenate(parts_idx)
        all_centers = np.concatenate(parts_ctr, axis=0)

        return (all_embeddings, all_indices, all_centers, np.array([]))

    # =========================================================================
    # Patch-level query encoding
    # =========================================================================

    def encode_query_unified(
        self,
        raw_rgb_masked: np.ndarray,
        mask: np.ndarray,
        raw_depth: np.ndarray,
        query_K: np.ndarray,
        median_template_depth: float,
    ):
        """
        Single DINO forward pass for both patch-level and global embeddings.

        RGBD mode: depth IS sampled. Only foreground grid cells with valid depth
        are kept. The third column of centers_depth is the actual depth in mm.

        Returns:
            embeddings:   (K, D) float32, L2-normalised patch features.
            centers_depth:(K, 3) float32, [u, v, depth_mm] in original image coords.
            global_emb:   (D,)  float32, L2-normalised global embedding.
        """
        D = self.dino_model.embed_dim
        h_orig, w_orig = raw_rgb_masked.shape[:2]
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return (
                np.zeros((0, D), dtype=np.float32),
                np.zeros((0, 3), dtype=np.float32),
                np.zeros(D, dtype=np.float32),
            )

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        max_dim = max(x_max - x_min, y_max - y_min)
        padding_ratio = self.configs.get("template_crop_padding_ratio", 0.2)
        pad = int(max_dim * padding_ratio)
        padded_side = max(max_dim + 2 * pad, 32)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        effective_side = padded_side

        rgb_crop, crop_x0, crop_y0 = extract_crop_with_coords(
            raw_rgb_masked, center_x, center_y, effective_side
        )
        mask_crop, _, _ = extract_crop_with_coords(
            mask, center_x, center_y, effective_side
        )

        sz = self.dino_input_size
        preprocess = T.Compose(
            [
                T.Resize((sz, sz), interpolation=T.InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        inp = preprocess(Image.fromarray(rgb_crop)).unsqueeze(0).cuda()

        use_intermediate = self.configs.get("dino_patch_use_intermediate_layer", False)
        n_blocks = len(self.dino_model.blocks)
        patch_layer_idx = (
            self.configs.get("dino_patch_intermediate_layer_idx", 18)
            if use_intermediate
            else n_blocks - 1
        )
        global_layer_idx = self.configs.get("dino_global_layer_idx", n_blocks - 1)
        patch_apply_norm = self.configs.get("dino_patch_apply_norm", False) or (
            not use_intermediate
        )
        global_apply_norm = self.configs.get("dino_global_apply_norm", False)
        use_patch_features = self.configs.get("high_level_use_patch_features", True)

        with torch.no_grad():
            if patch_layer_idx == global_layer_idx:
                outputs = self.dino_model.get_intermediate_layers(
                    inp,
                    n=[patch_layer_idx],
                    norm=False,
                    return_class_token=True,
                )
                patch_tokens = outputs[0][0].detach().float()
                global_tokens = outputs[0][0].detach().float()
                global_cls = outputs[0][1].detach().float()
            else:
                outputs = self.dino_model.get_intermediate_layers(
                    inp,
                    n=[patch_layer_idx, global_layer_idx],
                    norm=False,
                    return_class_token=True,
                )
                patch_tokens = outputs[0][0].detach().float()
                global_tokens = outputs[1][0].detach().float()
                global_cls = outputs[1][1].detach().float()

        if patch_apply_norm:
            patch_tokens = self.dino_model.norm(patch_tokens)
        if global_apply_norm:
            global_tokens = self.dino_model.norm(global_tokens)
            global_cls = self.dino_model.norm(global_cls)

        N_tok = patch_tokens.shape[1]
        feat_h = feat_w = int(N_tok**0.5)

        feat_2d = patch_tokens[0].reshape(feat_h, feat_w, D).detach().cpu().numpy()
        norms = np.linalg.norm(feat_2d, axis=-1, keepdims=True).clip(min=1e-6)
        # L2-normalise
        feat_2d = feat_2d / norms

        mask_small = cv2.resize(
            mask_crop.astype(np.uint8),
            (feat_w, feat_h),
            interpolation=cv2.INTER_NEAREST,
        )
        rows, cols = np.where(mask_small > 0)

        if len(rows) == 0:
            embeddings = np.zeros((0, D), dtype=np.float32)
            centers_depth = np.zeros((0, 3), dtype=np.float32)
        else:
            # (K, D)
            embeddings = feat_2d[rows, cols]

            u_crop = (cols + 0.5) / feat_w * effective_side
            v_crop = (rows + 0.5) / feat_h * effective_side
            u_orig = u_crop + crop_x0
            v_orig = v_crop + crop_y0

            # Sample depth from the full-resolution raw depth map
            ui = np.clip(u_orig.astype(int), 0, w_orig - 1)
            vi = np.clip(v_orig.astype(int), 0, h_orig - 1)
            depth_vals = raw_depth[vi, ui].astype(np.float32)

            erosion_kernel_size = self.configs.get("mask_erosion_kernel_size", 0)
            if erosion_kernel_size > 0:
                kernel = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
                eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)
                valid = (depth_vals > 0) & (eroded_mask[vi, ui] > 0)
            else:
                valid = (depth_vals > 0) & (mask[vi, ui] > 0)

            embeddings = embeddings[valid].astype(np.float32)
            u_orig = u_orig[valid]
            v_orig = v_orig[valid]
            depth_vals = depth_vals[valid]
            centers_depth = np.stack([u_orig, v_orig, depth_vals], axis=1).astype(
                np.float32
            )

        # Global embedding
        gt_2d = global_tokens[0].reshape(feat_h, feat_w, D).detach().cpu().numpy()
        fg = mask_small > 0
        if use_patch_features:
            global_emb = gt_2d[fg].mean(axis=0) if fg.any() else gt_2d.mean(axis=(0, 1))
        else:
            global_emb = global_cls[0].detach().cpu().numpy()

        emb_norm = np.linalg.norm(global_emb)
        if emb_norm > 1e-6:
            global_emb = global_emb / emb_norm

        return embeddings, centers_depth, global_emb.astype(np.float32)

    def encode_highlevel(
        self,
        rgb_np: Union[np.ndarray, torch.Tensor, List[np.ndarray]],
        mask_np: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    ) -> np.ndarray:
        """
        Encode with the DINO backbone for high-level (global) matching.

        If high_level_use_patch_features is enabled, uses masked average pooling
        of patch tokens (only foreground patches contribute). Otherwise uses CLS token.

        Args:
            rgb_np: RGB image(s) - HxWx3, BxHxWx3, or list of HxWx3
            mask_np: Optional foreground mask(s) - HxW, BxHxW, or list of HxW
                     When provided, used for accurate foreground detection in patch pooling.
        """
        use_patch_features = self.configs.get("high_level_use_patch_features", False)

        # Preprocess expects PIL images
        if isinstance(rgb_np, np.ndarray):
            if rgb_np.ndim == 3:
                # Single image: HxWx3
                pil_img = Image.fromarray(rgb_np)
                inp = self.dino_preprocess(pil_img).unsqueeze(0).to("cuda")
            elif rgb_np.ndim == 4:
                # Batch of images: BxHxWx3
                pil_imgs = [Image.fromarray(img) for img in rgb_np]
                inp = torch.stack(
                    [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
                ).to("cuda")
            else:
                raise ValueError("rgb_np must be HxWx3 or BxHxWx3 numpy array.")
        elif isinstance(rgb_np, list):
            # List of numpy arrays
            pil_imgs = [Image.fromarray(img) for img in rgb_np]
            inp = torch.stack(
                [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
            ).to("cuda")
        else:
            # Assume torch tensor input is in shape BxHxWx3
            if rgb_np.ndim == 3:
                # Add batch dimension
                rgb_np = rgb_np.unsqueeze(0)
            # Convert to list of PIL images
            rgb_np_cpu = rgb_np.cpu().numpy()
            pil_imgs = [Image.fromarray(img) for img in rgb_np_cpu]
            inp = torch.stack(
                [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
            ).to("cuda")

        # Process mask if provided
        mask_tensor = None
        if mask_np is not None and use_patch_features:
            # Resize mask to match the RGB preprocessing exactly:
            # torchvision Resize(hl_resize) → CenterCrop(hl), so we do the same.
            hl = self.dino_input_size
            # matches self.dino_preprocess
            hl_resize = round(hl * 256 / 224)
            offset = (hl_resize - hl) // 2

            def resize_mask_like_rgb(mask_t):
                # mask_t: (1, 1, H, W)
                m = F.interpolate(mask_t, size=(hl_resize, hl_resize), mode="nearest")
                return m[:, :, offset : offset + hl, offset : offset + hl]

            if isinstance(mask_np, np.ndarray):
                if mask_np.ndim == 2:
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0)
                    mask_tensor = resize_mask_like_rgb(mask_t).squeeze(1).to("cuda")
                elif mask_np.ndim == 3:
                    mask_t = torch.from_numpy(mask_np).float().unsqueeze(1)
                    mask_tensor = F.interpolate(
                        mask_t, size=(hl_resize, hl_resize), mode="nearest"
                    )
                    mask_tensor = mask_tensor[
                        :, :, offset : offset + hl, offset : offset + hl
                    ]
                    mask_tensor = mask_tensor.squeeze(1).to("cuda")
            elif isinstance(mask_np, list):
                resized_masks = []
                for m in mask_np:
                    mask_t = torch.from_numpy(m).float().unsqueeze(0).unsqueeze(0)
                    mask_resized = resize_mask_like_rgb(mask_t)
                    resized_masks.append(mask_resized.squeeze(0).squeeze(0))
                mask_tensor = torch.stack(resized_masks).to("cuda")

        with torch.no_grad():
            feat = self.masked_patch_pooling(inp, mask_tensor)

        feat = feat.detach().cpu().numpy()
        if feat.shape[0] == 1:
            # Remove batch dimension for single image
            feat = feat.squeeze(0)
        return feat

    def masked_patch_pooling(
        self, inp: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Extract DINO patch tokens and perform masked (optionally attention-weighted) pooling.

        Only foreground patches contribute to the final embedding.
        This makes the encoding robust to occlusion since occluded/background
        regions don't affect the embedding.

        Args:
            inp: Preprocessed input tensor [B, 3, 224, 224]
            mask: Optional foreground mask [B, 224, 224]. If provided, used for
                  accurate foreground detection. Otherwise falls back to
                  intensity-based detection.

        Returns:
            Masked-averaged (or attention-weighted) patch features [B, 768]
        """
        B = inp.shape[0]

        # Get patch tokens from the DINO backbone
        hl_inter_idx = self.configs.get("dino_global_layer_idx", None)
        global_apply_norm = self.configs.get("dino_global_apply_norm", False)

        outputs = self.dino_model.get_intermediate_layers(
            inp, n=[hl_inter_idx], norm=False, return_class_token=False
        )
        # (B, N, D)
        patch_tokens = outputs[0]

        if global_apply_norm:
            patch_tokens = self.dino_model.norm(patch_tokens)

        _, N, D = patch_tokens.shape
        H = W = int(N**0.5)

        # Reshape to spatial grid
        # [B, H, W, D]
        patch_tokens_spatial = patch_tokens.reshape(B, H, W, D)

        # Patch size for downsampling; inp is (B, C, H_img, W_img).
        patch_size = inp.shape[-1] // H

        if mask is not None:
            # Downsample the mask to patch grid, shape [B, H, W]; a patch
            # counts as foreground if over half its pixels are foreground.
            mask_downsampled = F.avg_pool2d(
                mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size
            ).squeeze(1)
            patch_mask = (mask_downsampled > 0.5).float()
        else:
            # Fallback: Create patch-level foreground mask from input intensity
            # Input is already segmented (background is black/near-zero after normalization)

            # Undo normalization approximately to detect foreground
            # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean = torch.tensor([0.485, 0.456, 0.406], device=inp.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=inp.device).view(
                1, 3, 1, 1
            )
            # Approximate [0, 1] range
            inp_unnorm = inp * std + mean

            # Compute per-pixel intensity (grayscale)
            # [B, 224, 224]
            inp_gray = inp_unnorm.mean(dim=1)

            # Downsample to patch grid size using average pooling
            patch_intensity = F.avg_pool2d(
                inp_gray.unsqueeze(1), kernel_size=patch_size, stride=patch_size
            )
            # [B, H, W]
            patch_intensity = patch_intensity.squeeze(1)

            # Threshold to get binary foreground mask
            fg_threshold = 0.02
            # [B, H, W]
            patch_mask = (patch_intensity > fg_threshold).float()

        # Ensure at least some patches are selected (fallback to all if none pass threshold)
        # [B, 1, 1]
        num_fg_patches = patch_mask.sum(dim=(1, 2), keepdim=True)
        # If no foreground patches, use all patches
        all_ones = torch.ones_like(patch_mask)
        # Expand condition to match patch_mask shape [B, H, W]
        has_fg = (num_fg_patches > 0).expand_as(patch_mask)
        patch_mask = torch.where(has_fg, patch_mask, all_ones)

        # Standard masked average pooling
        num_fg_patches = patch_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
        # [B, H, W, 1]
        patch_mask_expanded = patch_mask.unsqueeze(-1)
        # [B, H, W, D]
        masked_features = patch_tokens_spatial * patch_mask_expanded

        # Sum over spatial dimensions and normalize by foreground patch count
        # [B, D]
        feat_sum = masked_features.sum(dim=(1, 2))
        # [B, D]
        feat = feat_sum / num_fg_patches.squeeze(-1)

        # L2 normalize
        feat = F.normalize(feat, p=2, dim=-1)

        return feat.float()

    # =========================================================================
    # 3D-3D correspondence finding
    # =========================================================================

    def find_3d_2_3d_correspondences(
        self,
        nn_indices,
        template_data,
        query_data,
        num_neighbors=3,
        nn_distances=None,
        mnn_mask=None,
    ):
        """
        Build 3D-3D correspondences from DINO grid-cell nearest-neighbor matches.

        Uses grid-cell centers directly (no correlation refinement) — depth is stored
        in the third column of query_patches_centers_coors_depth.

        Args:
            nn_indices: (N, K) nearest neighbor indices for each query grid cell
            template_data: tuple of (template_indices, all_template_center_coor_values,
                             template_embeddings)
            query_data: tuple of (_, _, intrinsics, center_coords_depth, _)
            num_neighbors: top-K template matches to use per query grid cell
            nn_distances: (N, K) KNN distances; when provided, returned weights are
                          exp(-d) normalised to a probability vector for feature-aware
                          RANSAC sampling. None → uniform weights (ones / N).

        Returns:
            source_3d_points_on_object_model: list of (3,) arrays in object frame
            target_3d_points_on_query_object: list of (3,) arrays in camera frame
            weights: 1-D float32 array (sum=1) for RANSAC sampling
        """
        (template_indices, all_template_center_coor_values, tmpl_emb_all) = (
            template_data
        )
        (
            _,
            _,
            query_patches_intrinsics,
            query_patches_centers_coors_depth,
            _,
        ) = query_data

        target_3d_points_on_query_object = []
        source_3d_points_on_object_model = []
        # per-correspondence template embedding
        corr_template_embeddings = []
        raw_weights = []

        # Build correspondences using grid-cell centers directly
        for i in range(nn_indices.shape[0]):
            # Get query grid-cell center and intrinsics
            current_query_center_coords_depth = query_patches_centers_coors_depth[i]
            if hasattr(current_query_center_coords_depth, "tolist"):
                current_query_center_coords_depth = (
                    current_query_center_coords_depth.tolist()
                )
            # col
            u_q_center = current_query_center_coords_depth[0]
            # row
            v_q_center = current_query_center_coords_depth[1]
            current_query_patch_intrinsics = np.asarray(query_patches_intrinsics[i])

            # Process top-K nearest neighbors
            for neighbor_idx in range(num_neighbors):
                if mnn_mask is not None and not mnn_mask[i, neighbor_idx]:
                    continue
                neighbor_template_num = nn_indices[i][neighbor_idx]
                template_data_idx = template_indices[neighbor_template_num]
                base_template_idx = template_data_idx // self.num_inplane_variations

                template_patch_center_coor_value = all_template_center_coor_values[
                    neighbor_template_num
                ]
                base_template_pose = self.list_of_template_transformations[
                    base_template_idx
                ]

                u_q = u_q_center
                v_q = v_q_center

                # Depth is stored directly in the third column
                query_depth_value = float(current_query_center_coords_depth[2])

                if query_depth_value <= 0:
                    # Invalid depth
                    continue

                # Backproject refined query point to 3D
                fx_q, fy_q = (
                    current_query_patch_intrinsics[0, 0],
                    current_query_patch_intrinsics[1, 1],
                )
                cx_q, cy_q = (
                    current_query_patch_intrinsics[0, 2],
                    current_query_patch_intrinsics[1, 2],
                )

                pt_x_q = (u_q - cx_q) * query_depth_value / fx_q
                pt_y_q = (v_q - cy_q) * query_depth_value / fy_q
                pt_z_q = query_depth_value

                Y = np.array([pt_x_q, pt_y_q, pt_z_q], dtype=np.float32)

                # Backproject template center to 3D (no offset - we matched TO this point)
                inv_template_pose = np.linalg.inv(base_template_pose)
                template_patches_intrinsics = self.template_intrinsics

                u_t = template_patch_center_coor_value[0]
                v_t = template_patch_center_coor_value[1]
                template_depth_value = template_patch_center_coor_value[-1]

                fx_t, fy_t = (
                    template_patches_intrinsics[0, 0],
                    template_patches_intrinsics[1, 1],
                )
                cx_t, cy_t = (
                    template_patches_intrinsics[0, 2],
                    template_patches_intrinsics[1, 2],
                )

                # Invalid: missing depth (<=0) or the 16-bit background sentinel (>=60000).
                if template_depth_value <= 0 or template_depth_value >= 60000:
                    continue

                pt_X = (u_t - cx_t) * template_depth_value / fx_t
                pt_Y = (v_t - cy_t) * template_depth_value / fy_t
                pt_Z = template_depth_value

                X = np.array([pt_X, pt_Y, pt_Z], dtype=np.float32)

                # Transform to object model coordinate system
                X_on_object_model = (
                    np.dot(X, inv_template_pose[:3, :3].T) + inv_template_pose[:3, 3].T
                )

                # Store correspondence
                target_3d_points_on_query_object.append(Y)
                source_3d_points_on_object_model.append(X_on_object_model)
                corr_template_embeddings.append(
                    tmpl_emb_all[neighbor_template_num]
                )
                if nn_distances is not None:
                    raw_weights.append(nn_distances[i][neighbor_idx])

        assert len(source_3d_points_on_object_model) == len(
            target_3d_points_on_query_object
        ), "The number of source and target points should be equal."

        # Build sampling weight vector (lower KNN distance = higher weight)
        n_corr = len(source_3d_points_on_object_model)
        if nn_distances is not None and n_corr > 0:
            raw_w = np.array(raw_weights, dtype=np.float32)
            weights = np.exp(-raw_w)
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_corr, dtype=np.float32) / max(n_corr, 1)

        return (
            source_3d_points_on_object_model,
            target_3d_points_on_query_object,
            weights,
            corr_template_embeddings,
        )

    def find_3d_2_3d_correspondences_vectorized(
        self,
        nn_indices,
        template_data,
        query_data,
        num_neighbors=1,
        nn_distances=None,
        mnn_mask=None,
    ):
        """Vectorized drop-in replacement for find_3d_2_3d_correspondences.

        Replaces the Python double for-loop (N×K iterations) with numpy
        broadcasting. Uses self.inv_template_transformations (pre-cached in
        __init__) so np.linalg.inv is never called here.

        Returns identical types to the original method:
            source_3d: list of (3,) arrays in object frame
            target_3d: list of (3,) arrays in camera frame
            weights:   1-D float32 array (sum=1) for RANSAC sampling
            corr_template_embeddings: list of (D,) template embeddings aligned with source_3d
        """
        (template_indices, all_template_center_coor_values, tmpl_emb_all) = (
            template_data
        )
        (
            _,
            _,
            query_patches_intrinsics,
            query_patches_centers_coors_depth,
            _,
        ) = query_data

        K = num_neighbors
        # (N, 3)
        query_coors = np.asarray(
            query_patches_centers_coors_depth, dtype=np.float64
        )
        N = query_coors.shape[0]

        # ── Query backprojection ──────────────────────────────────────────────
        # query_patches_intrinsics may be (N,3,3) or list of arrays.
        K_arr = np.asarray(query_patches_intrinsics, dtype=np.float64)
        # (N,)
        fx_q = K_arr[:, 0, 0]
        fy_q = K_arr[:, 1, 1]
        cx_q = K_arr[:, 0, 2]
        cy_q = K_arr[:, 1, 2]

        u_q = query_coors[:, 0]
        v_q = query_coors[:, 1]
        d_q = query_coors[:, 2]

        # (N, 3)
        Y = np.stack(
            [(u_q - cx_q) * d_q / fx_q, (v_q - cy_q) * d_q / fy_q, d_q],
            axis=1,
        )

        # ── Flatten top-K template lookups ────────────────────────────────────
        # (N*K,)
        flat_nn = nn_indices[:, :K].ravel()

        # (N*K, 3+)
        tmpl_coors = np.asarray(all_template_center_coor_values, dtype=np.float64)[
            flat_nn
        ]

        # Template intrinsics (same for all templates)
        fx_t = self.template_intrinsics[0, 0]
        fy_t = self.template_intrinsics[1, 1]
        cx_t = self.template_intrinsics[0, 2]
        cy_t = self.template_intrinsics[1, 2]

        u_t = tmpl_coors[:, 0]
        v_t = tmpl_coors[:, 1]
        d_t = tmpl_coors[:, 2]

        # (N*K, 3)
        X_cam = np.stack(
            [(u_t - cx_t) * d_t / fx_t, (v_t - cy_t) * d_t / fy_t, d_t],
            axis=1,
        )

        # ── Batch transform to object frame ───────────────────────────────────
        tmpl_data_indices = np.asarray(template_indices, dtype=np.int64)[flat_nn]
        # (N*K,)
        base_tmpl_ids = tmpl_data_indices // self.num_inplane_variations

        # (N*K, 4, 4)
        inv_poses = self.inv_template_transformations[base_tmpl_ids]
        # (N*K, 3)
        X_obj = (
            np.einsum("nij,nj->ni", inv_poses[:, :3, :3], X_cam) + inv_poses[:, :3, 3]
        )

        # ── Validity mask ─────────────────────────────────────────────────────
        # (N*K,)
        d_q_rep = np.repeat(d_q, K)
        # Invalid: missing depth (<=0) or the 16-bit background sentinel (>=60000).
        valid = (d_q_rep > 0) & (d_t > 0) & (d_t < 60000)
        if mnn_mask is not None:
            # mnn_mask is (N, K_use); ravel to (N*K,) to match valid's shape.
            valid = valid & mnn_mask[:, :K].ravel()

        # (N*K, 3)
        Y_rep = np.repeat(Y, K, axis=0)

        source_3d = list(X_obj[valid].astype(np.float32))
        target_3d = list(Y_rep[valid].astype(np.float32))

        # Template embeddings aligned with valid correspondences, (M, D).
        tmpl_emb_arr = np.asarray(tmpl_emb_all, dtype=np.float32)
        corr_template_embeddings = list(tmpl_emb_arr[flat_nn][valid])

        # ── Weights ───────────────────────────────────────────────────────────
        n_corr = len(source_3d)
        if nn_distances is not None and n_corr > 0:
            raw_w = nn_distances[:, :K].ravel()[valid].astype(np.float32)
            weights = np.exp(-raw_w)
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_corr, dtype=np.float32) / max(n_corr, 1)

        return source_3d, target_3d, weights, corr_template_embeddings

    @staticmethod
    def compute_depth_centroid(
        depth_mm: np.ndarray,
        mask: np.ndarray,
        K: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Backproject masked depth pixels and return their 3D centroid in mm.

        Args:
            depth_mm: (H, W) float — depth map in millimetres.
            mask:     (H, W) uint8 — binary foreground mask.
            K:        (3, 3) float — camera intrinsics.

        Returns:
            (3,) float32 centroid [X, Y, Z] in camera frame (mm), or None if
            fewer than 10 valid depth pixels are found.
        """
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return None
        d = depth_mm[ys, xs].astype(np.float32)
        valid = d > 0
        if valid.sum() < 10:
            return None
        xs, ys, d = xs[valid], ys[valid], d[valid]
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        X = (xs - cx) * d / fx
        Y = (ys - cy) * d / fy
        return np.array([X.mean(), Y.mean(), d.mean()], dtype=np.float32)
