"""
correspondence.py
High-level template retrieval, patch-level DINO encoding, and 2D-3D
correspondence finding.

Query depth is never used for correspondences in this RGB pipeline: query
patch centers stay as 2D pixel coordinates, and only the cached template
patch depth (already backprojected to object-frame 3D at cache-build time)
supplies the 3D side of each correspondence.
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

    def encode_nearest_templates(self, top_nearest_templates: np.ndarray):
        """
        Return pre-computed DINO patch embeddings for the requested template views.

        Opens the H5 cache built by TemplateCache.build_caches() and reads only the
        requested tmpl_idx datasets — O(K) dict lookups, zero GPU work.
        """
        parts_emb, parts_idx, parts_ctr = [], [], []

        with h5py.File(self.template_cache.resolved_path, "r") as f:
            tmpls = f["templates"]
            for tmpl_idx in top_nearest_templates:
                key = str(int(tmpl_idx))
                if key not in tmpls:
                    continue
                emb = tmpls[key]["embeddings"][:]  # (K_i, D)
                ctr = tmpls[key]["centers"][:]  # (K_i, 3)
                parts_emb.append(emb)
                parts_idx.append(np.full(len(emb), tmpl_idx, dtype=np.int64))
                parts_ctr.append(ctr)

        if not parts_emb:
            D = self.dino_model.embed_dim
            return (
                np.zeros((0, D)),
                np.zeros(0, dtype=np.int64),
                np.zeros((0, 3)),
            )

        all_embeddings = np.concatenate(parts_emb, axis=0)
        all_indices = np.concatenate(parts_idx)
        all_centers = np.concatenate(parts_ctr, axis=0)

        return (all_embeddings, all_indices, all_centers)

    # =========================================================================
    # Patch-level query encoding
    # =========================================================================

    def encode_query_unified(
        self,
        raw_rgb_masked: np.ndarray,
        mask: np.ndarray,
    ):
        """
        Single DINO forward pass for both patch-level and global embeddings.

        RGB mode: depth is NOT sampled. All foreground grid cells are kept.
        The third column of centers_depth is set to 0.0 (sentinel) to preserve
        the (K, 3) shape expected by downstream code.
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
        global_layer_idx = self.configs.get("cache_v2_global_layer_idx", n_blocks - 1)
        patch_apply_norm = self.configs.get("cache_v2_patch_apply_norm", False) or (
            not use_intermediate
        )
        global_apply_norm = self.configs.get("cache_v2_global_apply_norm", False)
        use_patch_features = self.configs.get("high_level_use_patch_features", True)

        with torch.no_grad():
            if patch_layer_idx == global_layer_idx:
                outputs = self.dino_model.get_intermediate_layers(
                    inp,
                    n=[patch_layer_idx],
                    norm=False,
                    return_class_token=True,
                )
                patch_tokens = outputs[0][0].detach().float()  # (1, P, D)
                global_tokens = outputs[0][0].detach().float()  # (1, P, D)
                global_cls = outputs[0][1].detach().float()  # (1, D)
            else:
                outputs = self.dino_model.get_intermediate_layers(
                    inp,
                    n=[patch_layer_idx, global_layer_idx],
                    norm=False,
                    return_class_token=True,
                )
                patch_tokens = outputs[0][0].detach().float()  # (1, P, D)
                global_tokens = outputs[1][0].detach().float()  # (1, P, D)
                global_cls = outputs[1][1].detach().float()  # (1, D)

        if patch_apply_norm:
            patch_tokens = self.dino_model.norm(patch_tokens)
        if global_apply_norm:
            global_tokens = self.dino_model.norm(global_tokens)
            global_cls = self.dino_model.norm(global_cls)

        N_tok = patch_tokens.shape[1]
        feat_h = feat_w = int(N_tok**0.5)

        feat_2d = patch_tokens[0].reshape(feat_h, feat_w, D).detach().cpu().numpy()
        norms = np.linalg.norm(feat_2d, axis=-1, keepdims=True).clip(min=1e-6)
        feat_2d = feat_2d / norms  # L2-normalise

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
            embeddings = feat_2d[rows, cols]  # (K, D)

            u_crop = (cols + 0.5) / feat_w * effective_side
            v_crop = (rows + 0.5) / feat_h * effective_side
            u_orig = u_crop + crop_x0
            v_orig = v_crop + crop_y0

            # No depth sampling — all foreground cells valid; z=0 sentinel
            valid = (
                mask[
                    np.clip(v_orig.astype(int), 0, h_orig - 1),
                    np.clip(u_orig.astype(int), 0, w_orig - 1),
                ]
                > 0
            )

            embeddings = embeddings[valid].astype(np.float32)
            u_orig = u_orig[valid]
            v_orig = v_orig[valid]
            centers_depth = np.stack(
                [u_orig, v_orig, np.zeros(valid.sum(), dtype=np.float32)],
                axis=1,
            ).astype(np.float32)

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
        """Encode with the DINO backbone for high-level (global) matching.

        If high_level_use_patch_features is enabled, uses masked average pooling
        of patch tokens (only foreground patches contribute). Otherwise uses CLS token."""
        use_patch_features = self.configs.get("high_level_use_patch_features", False)

        if isinstance(rgb_np, np.ndarray):
            if rgb_np.ndim == 3:
                pil_img = Image.fromarray(rgb_np)
                inp = self.dino_preprocess(pil_img).unsqueeze(0).to("cuda")
            elif rgb_np.ndim == 4:
                pil_imgs = [Image.fromarray(img) for img in rgb_np]
                inp = torch.stack(
                    [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
                ).to("cuda")
            else:
                raise ValueError("rgb_np must be HxWx3 or BxHxWx3 numpy array.")
        elif isinstance(rgb_np, list):
            pil_imgs = [Image.fromarray(img) for img in rgb_np]
            inp = torch.stack(
                [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
            ).to("cuda")
        else:
            if rgb_np.ndim == 3:
                rgb_np = rgb_np.unsqueeze(0)
            rgb_np_cpu = rgb_np.detach().cpu().numpy()
            pil_imgs = [Image.fromarray(img) for img in rgb_np_cpu]
            inp = torch.stack(
                [self.dino_preprocess(pil_img) for pil_img in pil_imgs]
            ).to("cuda")

        mask_tensor = None
        if mask_np is not None and use_patch_features:
            hl = self.dino_input_size
            hl_resize = round(hl * 256 / 224)
            offset = (hl_resize - hl) // 2

            def resize_mask_like_rgb(mask_t):
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
            feat = feat.squeeze(0)
        return feat

    def masked_patch_pooling(
        self, inp: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Masked average pooling of DINO patch tokens."""
        B = inp.shape[0]
        hl_inter_idx = self.configs.get("cache_v2_global_layer_idx", None)
        global_apply_norm = self.configs.get("cache_v2_global_apply_norm", False)

        outputs = self.dino_model.get_intermediate_layers(
            inp, n=[hl_inter_idx], norm=False, return_class_token=False
        )
        patch_tokens = outputs[0].detach()  # (B, N, D)

        if global_apply_norm:
            patch_tokens = self.dino_model.norm(patch_tokens)

        _, N, D = patch_tokens.shape
        H = W = int(N**0.5)
        patch_tokens_spatial = patch_tokens.reshape(B, H, W, D)

        patch_size = inp.shape[-1] // H

        if mask is not None:
            mask_downsampled = F.avg_pool2d(
                mask.unsqueeze(1).float(), kernel_size=patch_size, stride=patch_size
            ).squeeze(1)
            patch_mask = (mask_downsampled > 0.2).float()
        else:
            mean = torch.tensor([0.485, 0.456, 0.406], device=inp.device).view(
                1, 3, 1, 1
            )
            std = torch.tensor([0.229, 0.224, 0.225], device=inp.device).view(
                1, 3, 1, 1
            )
            inp_unnorm = inp * std + mean
            inp_gray = inp_unnorm.mean(dim=1)
            patch_intensity = F.avg_pool2d(
                inp_gray.unsqueeze(1), kernel_size=patch_size, stride=patch_size
            ).squeeze(1)
            patch_mask = (patch_intensity > 0.02).float()

        num_fg_patches = patch_mask.sum(dim=(1, 2), keepdim=True)
        all_ones = torch.ones_like(patch_mask)
        has_fg = (num_fg_patches > 0).expand_as(patch_mask)
        patch_mask = torch.where(has_fg, patch_mask, all_ones)

        num_fg_patches = patch_mask.sum(dim=(1, 2), keepdim=True).clamp(min=1)
        patch_mask_expanded = patch_mask.unsqueeze(-1)
        masked_features = patch_tokens_spatial * patch_mask_expanded
        feat_sum = masked_features.sum(dim=(1, 2))
        feat = feat_sum / num_fg_patches.squeeze(-1)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat.float()

    # =========================================================================
    # 2D-3D correspondence finding
    # =========================================================================

    def find_2d_3d_correspondences(
        self,
        nn_indices,
        template_data,
        query_data,
        num_neighbors=3,
        nn_distances=None,
        mnn_mask=None,
    ):
        """
        Build 2D-3D correspondences for PnP.

        Query side  : (u_q, v_q) pixel coordinates — no depth backprojection.
        Template side: backproject (u_t, v_t, d_t) → camera frame → object frame.

        Returns 5-tuple: (source_3d, target_2d, weights, corr_template_embeddings,
                          raw_distances) where raw_distances are the raw L2 feature
                          distances per correspondence (for soft inlier scoring).
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

        target_2d_points = []
        source_3d_points_on_object_model = []
        corr_template_embeddings = []
        raw_weights = []

        for i in range(nn_indices.shape[0]):
            current_query_center_coords = query_patches_centers_coors_depth[i]
            if hasattr(current_query_center_coords, "tolist"):
                current_query_center_coords = current_query_center_coords.tolist()
            u_q = current_query_center_coords[0]
            v_q = current_query_center_coords[1]
            Y = np.array([u_q, v_q], dtype=np.float32)  # 2D target

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
                inv_template_pose = np.linalg.inv(base_template_pose)
                template_patches_intrinsics = self.template_intrinsics

                u_t = template_patch_center_coor_value[0]
                v_t = template_patch_center_coor_value[1]
                template_depth_value = template_patch_center_coor_value[-1]

                # Invalid: missing depth (<=0) or the 16-bit background sentinel (>=60000).
                if template_depth_value <= 0 or template_depth_value >= 60000:
                    continue

                fx_t = template_patches_intrinsics[0, 0]
                fy_t = template_patches_intrinsics[1, 1]
                cx_t = template_patches_intrinsics[0, 2]
                cy_t = template_patches_intrinsics[1, 2]

                pt_X = (u_t - cx_t) * template_depth_value / fx_t
                pt_Y = (v_t - cy_t) * template_depth_value / fy_t
                pt_Z = template_depth_value
                X = np.array([pt_X, pt_Y, pt_Z], dtype=np.float32)
                X_on_object_model = (
                    np.dot(X, inv_template_pose[:3, :3].T) + inv_template_pose[:3, 3].T
                )

                target_2d_points.append(Y)
                source_3d_points_on_object_model.append(X_on_object_model)
                corr_template_embeddings.append(tmpl_emb_all[neighbor_template_num])
                if nn_distances is not None:
                    raw_weights.append(nn_distances[i][neighbor_idx])

        assert len(source_3d_points_on_object_model) == len(target_2d_points)

        n_corr = len(source_3d_points_on_object_model)
        if nn_distances is not None and n_corr > 0:
            raw_w = np.array(raw_weights, dtype=np.float32)
            weights = np.exp(-raw_w)
            weights = weights / weights.sum()
        else:
            raw_w = np.zeros(n_corr, dtype=np.float32)
            weights = np.ones(n_corr, dtype=np.float32) / max(n_corr, 1)

        return (
            source_3d_points_on_object_model,
            target_2d_points,
            weights,
            corr_template_embeddings,
            raw_w,
        )

    def find_2d_3d_correspondences_vectorized(
        self,
        nn_indices,
        template_data,
        query_data,
        num_neighbors=1,
        nn_distances=None,
        mnn_mask=None,
    ):
        """
        Vectorised 2D-3D correspondence builder (drop-in for the loop version).

        Query targets are (u, v) pixel coordinates; template sources are
        3D object-frame points.

        Returns 5-tuple: (source_3d, target_2d, weights, corr_template_embeddings,
                          raw_distances) where raw_distances are the raw L2 feature
                          distances per correspondence (for soft inlier scoring).
        """
        (template_indices, all_template_center_coor_values, tmpl_emb_all) = (
            template_data
        )
        (
            _,
            _,
            _query_intrinsics,
            query_patches_centers_coors_depth,
            _,
        ) = query_data

        K_nn = num_neighbors
        query_coors = np.asarray(
            query_patches_centers_coors_depth, dtype=np.float64
        )  # (N, 3) — third column is 0 sentinel in RGB mode

        Y = query_coors[:, :2]  # (N, 2) — pixel (u, v) only

        flat_nn = nn_indices[:, :K_nn].ravel()  # (N*K,)
        tmpl_coors = np.asarray(all_template_center_coor_values, dtype=np.float64)[
            flat_nn
        ]  # (N*K, 3+)

        fx_t = self.template_intrinsics[0, 0]
        fy_t = self.template_intrinsics[1, 1]
        cx_t = self.template_intrinsics[0, 2]
        cy_t = self.template_intrinsics[1, 2]

        u_t = tmpl_coors[:, 0]
        v_t = tmpl_coors[:, 1]
        d_t = tmpl_coors[:, 2]

        X_cam = np.stack(
            [(u_t - cx_t) * d_t / fx_t, (v_t - cy_t) * d_t / fy_t, d_t],
            axis=1,
        )  # (N*K, 3)

        tmpl_data_indices = np.asarray(template_indices, dtype=np.int64)[flat_nn]
        base_tmpl_ids = tmpl_data_indices // self.num_inplane_variations
        inv_poses = self.inv_template_transformations[base_tmpl_ids]  # (N*K, 4, 4)
        X_obj = (
            np.einsum("nij,nj->ni", inv_poses[:, :3, :3], X_cam) + inv_poses[:, :3, 3]
        )  # (N*K, 3)

        # Only template depth required (no query depth check). Invalid: missing
        # depth (<=0) or the 16-bit background sentinel (>=60000).
        valid = (d_t > 0) & (d_t < 60000)
        if mnn_mask is not None:
            valid = valid & mnn_mask[:, :K_nn].ravel()

        Y_rep = np.repeat(Y, K_nn, axis=0)  # (N*K, 2)

        source_3d = list(X_obj[valid].astype(np.float32))
        target_2d = list(Y_rep[valid].astype(np.float32))

        tmpl_emb_arr = np.asarray(tmpl_emb_all, dtype=np.float32)
        corr_template_embeddings = list(tmpl_emb_arr[flat_nn][valid])

        n_corr = len(source_3d)
        if nn_distances is not None and n_corr > 0:
            raw_w = nn_distances[:, :K_nn].ravel()[valid].astype(np.float32)
            weights = np.exp(-raw_w)
            weights = weights / weights.sum()
        else:
            raw_w = np.zeros(n_corr, dtype=np.float32)
            weights = np.ones(n_corr, dtype=np.float32) / max(n_corr, 1)

        return source_3d, target_2d, weights, corr_template_embeddings, raw_w
