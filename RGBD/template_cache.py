"""
Builds and loads the on-disk template embedding cache for one object: an H5 file
of per-template patch embeddings (used for patch-level correspondence matching)
plus a global .npy array (used for high-level template retrieval). Both are
produced in a single DINO forward pass per batch.
"""

import os
import time

import cv2
import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from image_patch_utils import extract_crop_with_coords, rotate_image


class TemplateCropDataset(torch.utils.data.Dataset):
    """Dataset that performs all CPU-side template loading/cropping in worker processes."""

    def __init__(
        self, all_tmpl_meta, templates_dir, obj_id, padding_ratio, use_new_templates
    ):
        self.all_tmpl_meta = all_tmpl_meta
        self.templates_dir = templates_dir
        self.obj_id = obj_id
        self.padding_ratio = padding_ratio
        self.use_new_templates = use_new_templates

    def __len__(self):
        return len(self.all_tmpl_meta)

    def __getitem__(self, idx):
        tmpl_idx, base_view_id, inplane_rot_deg = self.all_tmpl_meta[idx]
        obj_dir = os.path.join(self.templates_dir, f"{self.obj_id:06d}")
        if self.use_new_templates:
            tmpl_path = os.path.join(
                obj_dir, "train_pbr/000000/rgb", f"{base_view_id:06d}.png"
            )
            mask_path = os.path.join(
                obj_dir, "train_pbr/000000/mask_visib", f"{base_view_id:06d}_000000.png"
            )
            depth_path = os.path.join(
                obj_dir, "train_pbr/000000/depth", f"{base_view_id:06d}.png"
            )
        else:
            tmpl_path = os.path.join(obj_dir, "rgb", f"{base_view_id:06d}.png")
            mask_path = os.path.join(
                obj_dir, "mask_visib", f"{base_view_id:06d}_000000.png"
            )
            depth_path = os.path.join(obj_dir, "depth", f"{base_view_id:06d}.png")

        tmpl_rgb = np.array(Image.open(tmpl_path).convert("RGB"))
        mask = np.where(np.array(Image.open(mask_path)) > 0, 1, 0).astype(np.uint8)
        depth = np.array(Image.open(depth_path), dtype=np.float32)

        if inplane_rot_deg != 0:
            rotated_rgb = rotate_image(tmpl_rgb, -inplane_rot_deg)
            rotated_mask = rotate_image(mask, -inplane_rot_deg)
            rotated_mask = np.where(rotated_mask > 0, 1, 0).astype(np.uint8)
        else:
            rotated_rgb = tmpl_rgb
            rotated_mask = mask

        h_orig, w_orig = rotated_rgb.shape[:2]
        rows_m, cols_m = np.where(rotated_mask > 0)
        if len(rows_m) == 0:
            return None

        x_min, x_max = int(cols_m.min()), int(cols_m.max())
        y_min, y_max = int(rows_m.min()), int(rows_m.max())
        max_dim = max(x_max - x_min, y_max - y_min)
        pad = int(max_dim * self.padding_ratio)
        crop_side = max(max_dim + 2 * pad, 32)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        tmpl_rgb_crop, crop_x0, crop_y0 = extract_crop_with_coords(
            rotated_rgb, center_x, center_y, crop_side
        )
        tmpl_mask_crop, _, _ = extract_crop_with_coords(
            rotated_mask, center_x, center_y, crop_side
        )

        return dict(
            tmpl_idx=tmpl_idx,
            crop_rgb=tmpl_rgb_crop,
            tmpl_mask_crop=tmpl_mask_crop,
            crop_side=crop_side,
            crop_x0=crop_x0,
            crop_y0=crop_y0,
            inplane_rot_deg=inplane_rot_deg,
            h_orig=h_orig,
            w_orig=w_orig,
            tmpl_depth=depth,
        )


class TemplateCache:
    """Construct once per object after the DINO model loads, then call ensure()."""

    def __init__(
        self,
        configs: dict,
        obj_id: int,
        dino_model,
        dino_input_size: int,
        storage_full_path: str,
        num_inplane_variations: int,
        inplane_rotation_angle: int,
    ):
        self.configs = configs
        self.obj_id = obj_id
        self.dino_model = dino_model
        self.dino_input_size = dino_input_size
        self.storage_full_path = storage_full_path
        self.num_inplane_variations = num_inplane_variations
        self.inplane_rotation_angle = inplane_rotation_angle
        # built lazily on first cache build
        self.template_preprocess = None
        # set by ensure()
        self.resolved_path = None

    def patch_cache_path(self) -> str:
        """Return path to the per-object H5 template patch embedding cache."""
        return os.path.join(self.storage_full_path, "template_patch_cache.h5")

    def is_valid(self, cache_path: str) -> bool:
        if not os.path.exists(cache_path):
            return False
        try:
            with h5py.File(cache_path, "r") as f:
                if "metadata" not in f:
                    return False
                meta = f["metadata"]
                checks = [
                    meta.attrs.get("caching_complete") == 1,
                    meta.attrs.get("templates_dir")
                    == self.template_dir_fingerprint(self.configs["templates_dir"]),
                    meta.attrs.get("inplane_rotation_angle")
                    == self.configs["inplane_rotation_angle"],
                    meta.attrs.get("template_crop_padding_ratio")
                    == float(self.configs.get("template_crop_padding_ratio", 0.2)),
                    meta.attrs.get("dino_input_size") == self.dino_input_size,
                    meta.attrs.get("dino_patch_layers")
                    == str(self.configs.get("dino_patch_layers", None)),
                    meta.attrs.get("dino_use_intermediate")
                    == int(
                        self.configs.get("dino_patch_use_intermediate_layer", False)
                    ),
                    meta.attrs.get("dino_intermediate_idx")
                    == int(
                        self.configs.get("dino_patch_intermediate_layer_idx", -1) or -1
                    ),
                ]
                return all(checks)
        except Exception:
            return False

    def build_caches(self, patch_path: str, global_path: str):
        """
        Single-pass template cache build: one DINO forward call per batch produces
        both the H5 patch cache and the global (high-level) embedding array.
        """
        use_new_templates = "new_templates" in self.configs["templates_dir"]
        if use_new_templates:
            sample_rgb_dir = os.path.join(
                self.configs["templates_dir"],
                f"{self.obj_id:06d}",
                "train_pbr/000000/rgb",
            )
        else:
            sample_rgb_dir = os.path.join(
                self.configs["templates_dir"],
                f"{self.obj_id:06d}",
                "rgb",
            )
        num_base_views = len(os.listdir(sample_rgb_dir))
        total = num_base_views * self.num_inplane_variations

        all_tmpl_meta = []
        for tmpl_idx in range(total):
            base_view_id = tmpl_idx // self.num_inplane_variations
            inplane_rot_deg = (
                tmpl_idx % self.num_inplane_variations
            ) * self.inplane_rotation_angle
            all_tmpl_meta.append((tmpl_idx, base_view_id, inplane_rot_deg))

        # Config knobs
        patch_layer_idx = self.configs.get("dino_patch_intermediate_layer_idx", 18)
        n_blocks = len(self.dino_model.blocks)
        global_layer_idx = self.configs.get("dino_global_layer_idx", n_blocks - 1)
        patch_apply_norm = self.configs.get("dino_patch_apply_norm", False)
        global_apply_norm = self.configs.get("dino_global_apply_norm", False)
        use_patch_features = self.configs.get("high_level_use_patch_features", True)

        batch_size = self.configs.get("dino_template_encode_batch_size", 16)
        num_workers = self.configs.get("dino_cache_num_workers", 4)

        print(
            f"[TemplateCache] Building both caches in a single pass: {total} templates, "
            f"batch={batch_size}, patch_layer={patch_layer_idx}, global_layer={global_layer_idx}"
        )

        # Crops are already square, so a plain resize suffices (no center-crop).
        sz = self.dino_input_size
        if self.template_preprocess is None:
            self.template_preprocess = T.Compose(
                [
                    T.Resize((sz, sz), interpolation=T.InterpolationMode.BICUBIC),
                    T.ToTensor(),
                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )
        preprocess = self.template_preprocess

        dataset = TemplateCropDataset(
            all_tmpl_meta=all_tmpl_meta,
            templates_dir=self.configs["templates_dir"],
            obj_id=self.obj_id,
            padding_ratio=float(self.configs.get("template_crop_padding_ratio", 0.2)),
            use_new_templates=use_new_templates,
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=lambda batch: batch,
            prefetch_factor=2 if num_workers > 0 else None,
            persistent_workers=num_workers > 0,
        )

        # Accumulate global embeddings as (tmpl_idx, emb) pairs
        # list of (int, np.ndarray D)
        global_pairs = []

        with h5py.File(patch_path, "w") as f:
            # Write metadata (same fields is_valid() checks)
            meta = f.create_group("metadata")
            meta.attrs["templates_dir"] = self.template_dir_fingerprint(
                self.configs["templates_dir"]
            )
            meta.attrs["inplane_rotation_angle"] = self.configs[
                "inplane_rotation_angle"
            ]
            meta.attrs["template_crop_padding_ratio"] = float(
                self.configs.get("template_crop_padding_ratio", 0.2)
            )
            meta.attrs["dino_input_size"] = self.dino_input_size
            meta.attrs["dino_patch_layers"] = str(
                self.configs.get("dino_patch_layers", None)
            )
            meta.attrs["dino_use_intermediate"] = int(
                self.configs.get("dino_patch_use_intermediate_layer", False)
            )
            meta.attrs["dino_intermediate_idx"] = int(
                self.configs.get("dino_patch_intermediate_layer_idx", -1) or -1
            )
            meta.attrs["caching_complete"] = 0

            tmpls_grp = f.create_group("templates")

            t_wait_total = 0.0
            t_gpu_total = 0.0
            t_post_total = 0.0
            n_batches = 0
            t_iter_start = time.perf_counter()

            for batch in tqdm(loader, desc="[TemplateCache] Single-pass encoding"):
                t_got_batch = time.perf_counter()
                t_wait_total += t_got_batch - t_iter_start

                valid_items = [item for item in batch if item is not None]
                if not valid_items:
                    t_iter_start = time.perf_counter()
                    continue

                valid_crops = [item["crop_rgb"] for item in valid_items]

                # ── Single forward pass: extract both patch and global layers ──
                tensors = [preprocess(Image.fromarray(img)) for img in valid_crops]
                inp = torch.stack(tensors, dim=0).cuda()

                t_gpu_start = time.perf_counter()
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                    if patch_layer_idx == global_layer_idx:
                        outputs = self.dino_model.get_intermediate_layers(
                            inp,
                            n=[patch_layer_idx],
                            norm=False,
                            return_class_token=True,
                        )
                        # (B, P, D)
                        patch_tokens = outputs[0][0].detach().float()
                        # (B, P, D)
                        global_tokens = outputs[0][0].detach().float()
                        # (B, D)
                        global_cls = outputs[0][1].detach().float()
                    else:
                        outputs = self.dino_model.get_intermediate_layers(
                            inp,
                            n=[patch_layer_idx, global_layer_idx],
                            norm=False,
                            return_class_token=True,
                        )
                        # (B, P, D)
                        patch_tokens = outputs[0][0].detach().float()
                        # (B, P, D)
                        global_tokens = outputs[1][0].detach().float()
                        # (B, D)
                        global_cls = outputs[1][1].detach().float()

                # Optional LayerNorm
                if patch_apply_norm:
                    patch_tokens = self.dino_model.norm(patch_tokens)
                if global_apply_norm:
                    global_tokens = self.dino_model.norm(global_tokens)
                    global_cls = self.dino_model.norm(global_cls)

                torch.cuda.synchronize()
                t_gpu_end = time.perf_counter()
                t_gpu_total += t_gpu_end - t_gpu_start

                t_post_start = time.perf_counter()

                N_batch, n_patches, D = patch_tokens.shape
                feat_h = feat_w = int(n_patches**0.5)

                for b, m in enumerate(valid_items):
                    crop_side = m["crop_side"]
                    crop_x0 = m["crop_x0"]
                    crop_y0 = m["crop_y0"]
                    inplane_rot_deg = m["inplane_rot_deg"]
                    h_orig, w_orig = m["h_orig"], m["w_orig"]
                    tmpl_depth = m["tmpl_depth"]
                    tmpl_idx = m["tmpl_idx"]

                    # ── Patch cache entry ──
                    feat_2d = patch_tokens[b].reshape(feat_h, feat_w, D).cpu().numpy()
                    norms = np.linalg.norm(feat_2d, axis=-1, keepdims=True).clip(
                        min=1e-6
                    )
                    # L2-normalise
                    feat_2d = feat_2d / norms

                    mask_small = cv2.resize(
                        m["tmpl_mask_crop"],
                        (feat_w, feat_h),
                        interpolation=cv2.INTER_NEAREST,
                    )
                    rows, cols = np.where(mask_small > 0)
                    if len(rows) == 0:
                        # Still try to store global embedding
                        pass
                    else:
                        cell_embeddings = feat_2d[rows, cols]

                        u_rot = (cols + 0.5) / feat_w * crop_side + crop_x0
                        v_rot = (rows + 0.5) / feat_h * crop_side + crop_y0

                        if inplane_rot_deg != 0:
                            inv_rot_mat = cv2.getRotationMatrix2D(
                                (w_orig / 2.0, h_orig / 2.0), inplane_rot_deg, 1.0
                            )
                            pts_hom = np.stack(
                                [u_rot, v_rot, np.ones_like(u_rot)], axis=1
                            )
                            u_orig_pts = (inv_rot_mat[0] * pts_hom).sum(axis=1)
                            v_orig_pts = (inv_rot_mat[1] * pts_hom).sum(axis=1)
                        else:
                            u_orig_pts = u_rot
                            v_orig_pts = v_rot

                        ui = np.clip(u_orig_pts.astype(int), 0, w_orig - 1)
                        vi = np.clip(v_orig_pts.astype(int), 0, h_orig - 1)
                        depth_vals = tmpl_depth[vi, ui].astype(np.float32)

                        # Invalid: missing depth (<=0) or the 16-bit background sentinel (>=60000).
                        valid_depth = (depth_vals > 0) & (depth_vals < 60000)
                        if np.any(valid_depth):
                            embeddings_out = cell_embeddings[valid_depth].astype(
                                np.float32
                            )
                            centers_out = np.stack(
                                [
                                    u_orig_pts[valid_depth],
                                    v_orig_pts[valid_depth],
                                    depth_vals[valid_depth],
                                ],
                                axis=1,
                            ).astype(np.float32)

                            grp = tmpls_grp.create_group(str(tmpl_idx))
                            grp.create_dataset(
                                "embeddings", data=embeddings_out, compression="lzf"
                            )
                            grp.create_dataset(
                                "centers", data=centers_out, compression="lzf"
                            )

                    # ── Global embedding ──
                    if use_patch_features:
                        gt_2d = (
                            global_tokens[b].reshape(feat_h, feat_w, D).cpu().numpy()
                        )
                        fg = mask_small > 0
                        if fg.any():
                            global_emb = gt_2d[fg].mean(axis=0)
                        else:
                            global_emb = gt_2d.mean(axis=(0, 1))
                    else:
                        global_emb = global_cls[b].cpu().numpy()

                    # L2-normalise
                    emb_norm = np.linalg.norm(global_emb)
                    if emb_norm > 1e-6:
                        global_emb = global_emb / emb_norm

                    global_pairs.append((int(tmpl_idx), global_emb.astype(np.float32)))

                t_post_total += time.perf_counter() - t_post_start
                n_batches += 1
                t_iter_start = time.perf_counter()

            t_total = t_wait_total + t_gpu_total + t_post_total
            print(
                f"[TemplateCache] Timing over {n_batches} batches  (total {t_total:.1f}s)\n"
                f"  DataLoader wait (CPU/IO): {t_wait_total:.1f}s  "
                f"({100*t_wait_total/max(t_total,1e-9):.0f}%)\n"
                f"  DINO forward    (GPU):   {t_gpu_total:.1f}s  "
                f"({100*t_gpu_total/max(t_total,1e-9):.0f}%)\n"
                f"  Post-process+H5 (CPU):   {t_post_total:.1f}s  "
                f"({100*t_post_total/max(t_total,1e-9):.0f}%)"
            )
            meta.attrs["caching_complete"] = 1

        print(f"[TemplateCache] Patch cache done. Saved to {patch_path}")

        # Build and save global embedding array
        if global_pairs:
            D = global_pairs[0][1].shape[0]
            global_array = np.zeros((total, D), dtype=np.float32)
            for tmpl_idx, emb in global_pairs:
                if tmpl_idx < total:
                    global_array[tmpl_idx] = emb
            np.save(global_path, global_array)
            print(f"[TemplateCache] Global embeddings done. Saved to {global_path}")
        else:
            print("[TemplateCache] Warning: no global embeddings collected.")

    def ensure(self) -> str:
        """Builds both caches if needed, resolves to local storage, returns the patch-cache path."""
        patch_path = self.patch_cache_path()
        global_path = f"{self.storage_full_path}/template_embeddings_high_level.npy"

        patch_valid = self.is_valid(patch_path)
        global_valid = os.path.exists(global_path)

        if patch_valid and global_valid:
            print("[TemplateCache] Both caches already valid — skipping build.")
        else:
            self.build_caches(patch_path, global_path)

        local_dir = self.local_cache_dir()
        if local_dir:
            import shutil

            local_h5 = os.path.join(local_dir, "template_patch_cache.h5")
            if not os.path.exists(local_h5):
                os.makedirs(local_dir, exist_ok=True)
                print(f"[Cluster] Copying H5 cache to node storage: {local_h5}")
                shutil.copy2(patch_path, local_h5)
            self.resolved_path = local_h5
        else:
            self.resolved_path = patch_path

        return self.resolved_path

    def load_high_level_embeddings(self):
        """Load global embeddings pre-built by build_caches()."""
        persistent_path = f"{self.storage_full_path}/template_embeddings_high_level.npy"
        if not os.path.exists(persistent_path):
            raise FileNotFoundError(
                f"[TemplateCache] Global embeddings missing: {persistent_path}"
            )

        local_dir = self.local_cache_dir()
        if local_dir:
            import shutil

            local_npy = os.path.join(local_dir, "template_embeddings_high_level.npy")
            if not os.path.exists(local_npy):
                os.makedirs(local_dir, exist_ok=True)
                print(f"[Cluster] Copying .npy cache to node storage: {local_npy}")
                shutil.copy2(persistent_path, local_npy)
            load_path = local_npy
        else:
            load_path = persistent_path

        print("Loading saved embeddings ...")
        return np.load(load_path)

    def local_cache_dir(self) -> str:
        """Return the $TMPDISK-local cache directory for this object, or empty string."""
        tmpdisk = os.environ.get("TMPDISK", "")
        if not tmpdisk or not os.getenv("SLURM_JOB_ID"):
            return ""
        return os.path.join(tmpdisk, "b2tf_cache", self.storage_full_path)

    @staticmethod
    def template_dir_fingerprint(path: str) -> str:
        """Return a stable suffix of the templates path, stripping volatile prefixes
        like $TMPDISK (which changes every SLURM job). Anchors on the first path
        component that looks like a dataset root (contains 'tudl' or 'templates')."""
        parts = path.rstrip("/").split("/")
        for anchor in ("tudl", "templates", "new_templates"):
            try:
                idx = next(i for i, p in enumerate(parts) if anchor in p)
                return "/".join(parts[idx:])
            except StopIteration:
                continue
        return parts[-1]
