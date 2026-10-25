"""
image_patch_utils.py
2D image/patch operations: cropping, rotation, and patch extraction/
generation around object masks. For 3D/depth geometry (backprojection,
pointclouds, pose error), see geometry_3d_utils.py.
"""

import cv2
import numpy as np
import torch
from scipy.ndimage import gaussian_filter
from typing import Union


def crop_bbox_square_centered(
    image: np.ndarray,
    mask: np.ndarray,
    padding_ratio: float = 0.2,
    fill_value: int = 0,
    return_mask: bool = False,
) -> Union[np.ndarray, tuple]:
    """
    Crop to square bbox with object centered, zero-padded if needed.

    This eliminates aspect ratio distortion and ensures consistent object positioning.

    Args:
        image: Input image (HxWx3 or HxWx4)
        mask: Binary mask indicating object region
        padding_ratio: Padding around object as fraction of bbox size
        fill_value: Value to use for padding
        return_mask: If True, also return the cropped mask

    Returns:
        If return_mask=False: cropped image
        If return_mask=True: (cropped_image, cropped_mask)
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        if return_mask:
            return image, mask
        return image
    img_h, img_w = image.shape[:2]
    num_channels = image.shape[2] if len(image.shape) == 3 else 1
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    bbox_w = x_max - x_min
    bbox_h = y_max - y_min
    max_dim = max(bbox_w, bbox_h)
    pad = int(max_dim * padding_ratio)
    square_size = max_dim + 2 * pad
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    half_size = square_size // 2
    crop_x0 = center_x - half_size
    crop_x1 = center_x + half_size
    crop_y0 = center_y - half_size
    crop_y1 = center_y + half_size
    canvas = np.full(
        (square_size, square_size, num_channels), fill_value, dtype=image.dtype
    )
    img_x0 = max(0, crop_x0)
    img_x1 = min(img_w, crop_x1)
    img_y0 = max(0, crop_y0)
    img_y1 = min(img_h, crop_y1)
    img_region = image[img_y0:img_y1, img_x0:img_x1]
    canvas_x0 = img_x0 - crop_x0
    canvas_y0 = img_y0 - crop_y0
    canvas_x1 = canvas_x0 + img_region.shape[1]
    canvas_y1 = canvas_y0 + img_region.shape[0]
    canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = img_region
    if num_channels == 4:
        canvas = canvas[:, :, :3]
    if return_mask:
        mask_canvas = np.zeros((square_size, square_size), dtype=mask.dtype)
        mask_region = mask[img_y0:img_y1, img_x0:img_x1]
        mask_canvas[canvas_y0:canvas_y1, canvas_x0:canvas_x1] = mask_region
        return canvas, mask_canvas
    return canvas

def extract_query_patches(
    scene_rgb: np.ndarray,
    scene_mask: np.ndarray,
    patch_size: int = 32,
    num_points: int = 30,
):
    h, w = scene_mask.shape

    centers = find_patch_center_coordinates(
        scene_rgb, scene_mask, num_points=num_points, edge_biased=True
    )
    patches = []

    for cx, cy in centers:
        x1, x2 = cx - patch_size // 2, cx + patch_size // 2
        y1, y2 = cy - patch_size // 2, cy + patch_size // 2

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        patch = scene_rgb[y1:y2, x1:x2, :]
        patch_tensor = torch.tensor(patch).permute(2, 0, 1)
        patches.append(patch_tensor)
    return patches, centers


def rotate_image_mask_points(
    image: np.ndarray, mask: np.ndarray, depth, angle: float, points=None
):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )
    rotated_mask = cv2.warpAffine(
        mask, rot_mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
    )

    rotated_depth = cv2.warpAffine(
        depth,
        rot_mat,
        (depth.shape[1], depth.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )
    transformed_points = []
    for point in points:
        x, y = point
        transformed_point = np.dot(rot_mat, np.array([x, y, 1]))
        x, y = transformed_point[:2]
        transformed_point = (int(x), int(y))
        transformed_points.append(transformed_point)

    return rotated_image, rotated_mask, rotated_depth, transformed_points


def extract_patches_at_centers(
    scene_rgb: np.ndarray,
    scene_mask: np.ndarray,
    scene_depth: np.ndarray,
    patch_center_coords: tuple,
    patch_size: int = 48,
):
    h, w = scene_mask.shape

    patches = []
    patches_masks = []
    patches_depths = []

    for coord in patch_center_coords:
        cx, cy = coord
        half_patch_size = patch_size // 2

        x1, x2 = cx - half_patch_size, cx + half_patch_size
        y1, y2 = cy - half_patch_size, cy + half_patch_size

        patch = scene_rgb[y1:y2, x1:x2, :]
        patch_mask = scene_mask[y1:y2, x1:x2]
        patch_depth = scene_depth[y1:y2, x1:x2]

        # Normalizing the patch to be in the range [0, 1]
        patch = patch.astype(np.float32) / 255.0
        patch_tensor = torch.tensor(patch).permute(2, 0, 1).float()
        patch_mask_tensor = torch.as_tensor(patch_mask).float()
        patch_depth_tensor = torch.as_tensor(patch_depth).float()
        patches.append(patch_tensor)
        patches_masks.append(patch_mask_tensor)
        patches_depths.append(patch_depth_tensor)

    return patches, patches_masks, patches_depths


def extract_patches_random_sampling(
    scene_rgb: np.ndarray,
    scene_mask: np.ndarray,
    scene_depth: np.ndarray,
    patch_size: int = 32,
    num_points: int = 30,
):
    h, w = scene_mask.shape

    centers = find_patch_center_coordinates(
        scene_rgb, scene_mask, scene_depth, num_points=num_points, edge_biased=True
    )
    patches = []
    patches_masks = []
    patches_depths = []
    patches_center_coordinates = []

    for cx, cy in centers:
        x1, x2 = cx - patch_size // 2, cx + patch_size // 2
        y1, y2 = cy - patch_size // 2, cy + patch_size // 2

        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            continue

        patch = scene_rgb[y1:y2, x1:x2, :]
        patch_mask = scene_mask[y1:y2, x1:x2]

        patch_tensor = torch.tensor(patch).permute(2, 0, 1)
        patch_mask_tensor = torch.as_tensor(patch_mask)

        patches.append(patch_tensor)
        patches_masks.append(patch_mask_tensor)
        # patches_center_coordinates.append((cx, cy))
        patches_center_coordinates.append((cy, cx))

        patch_depth = scene_depth[y1:y2, x1:x2]
        patch_depth_tensor = torch.as_tensor(patch_depth)
        patches_depths.append(patch_depth_tensor)

    return patches, patches_masks, patches_depths, patches_center_coordinates


def find_patch_center_coordinates(
    scene_rgb: np.ndarray,
    scene_mask: np.ndarray,
    scene_depth: np.ndarray,
    num_points: int = 30,
    edge_biased: bool = True,
    min_distance: float = 4.0,
    limit: int = 500,
):
    # # converting the mask to binary
    # scene_mask = np.where(scene_mask != 0, 1, 0)

    if edge_biased:
        gray = cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 100, 200)
        edges_with_rgb = edges * scene_mask
        edges_with_mask = cv2.Canny(scene_mask.astype(np.uint8) * 255, 100, 200)
        edges_combined = np.logical_or(edges_with_rgb, edges_with_mask)

        weighted_map = edges_combined.astype(np.float32) + 0.1 * scene_mask.astype(
            np.float32
        )
        weighted_map = gaussian_filter(weighted_map, sigma=1)
        # Zeroing out pixels where depth is zero
        depth_zero_pixels = np.where(scene_depth == 0)
        weighted_map[depth_zero_pixels] = 0
        weighted_map /= weighted_map.sum()

        flat_probs = weighted_map.flatten()

        number_of_candidates = min(limit, np.nonzero(flat_probs)[0].shape[0])
        choices = np.random.choice(
            np.arange(flat_probs.size), size=number_of_candidates, replace=False, p=flat_probs
        )

    else:
        mask_weighted_probs = scene_mask.copy().astype(np.float32)
        # Zeroing out pixels where depth is zero
        depth_zero_pixels = np.where(scene_depth == 0)
        mask_weighted_probs[depth_zero_pixels] = 0
        if mask_weighted_probs.sum() == 0:
            return []
        mask_weighted_probs /= mask_weighted_probs.sum()

        flat_probs = mask_weighted_probs.flatten()
        if np.nonzero(flat_probs)[0].shape[0] < 4:
            return []
        number_of_candidates = min(limit, np.nonzero(flat_probs)[0].shape[0])
        choices = np.random.choice(
            np.arange(flat_probs.size),
            size=number_of_candidates,
            replace=False,
            p=flat_probs,
        )

    ys, xs = np.unravel_index(choices, scene_mask.shape)
    candidates = list(zip(xs, ys))

    centers = []
    for pt in candidates:
        if all(
            np.linalg.norm(np.array(pt) - np.array(other)) >= min_distance
            for other in centers
        ):
            centers.append(pt)

    return centers


def rotate_image_mask_depth(
    image: np.ndarray, mask: np.ndarray, depth: np.ndarray, angle: float
):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )
    rotated_mask = cv2.warpAffine(
        mask, rot_mat, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST
    )
    rotated_depth = cv2.warpAffine(
        depth,
        rot_mat,
        (depth.shape[1], depth.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderValue=0,
    )

    return rotated_image, rotated_mask, rotated_depth


def erode_mask(mask: np.ndarray, erosion_size: int = 5) -> np.ndarray:
    """
    Erode a binary mask to shrink it inward by 'erosion_size' pixels.
    """
    kernel = np.ones((erosion_size, erosion_size), np.uint8)
    eroded_mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1)

    return eroded_mask


def rotate_image(image: np.ndarray, angle: float):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )

    return rotated_image


def extract_crop_with_coords(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    side: int,
    fill_value: int = 0,
) -> tuple:
    """
    Extract a square crop of size `side` centered at (center_x, center_y).

    Zero-pads at image boundaries using `fill_value`. Unlike
    `crop_bbox_square_centered`, the crop center and size are given explicitly
    rather than derived from a mask's bounding box.

    Returns:
        crop:     (side, side) or (side, side, C) numpy array.
        crop_x0:  x coordinate in the original image of the left edge of the crop.
        crop_y0:  y coordinate in the original image of the top edge of the crop.
    """
    h, w = image.shape[:2]
    half = side // 2
    crop_x0 = int(round(center_x)) - half
    crop_y0 = int(round(center_y)) - half
    crop_x1 = crop_x0 + side
    crop_y1 = crop_y0 + side

    if image.ndim == 3:
        crop = np.full((side, side, image.shape[2]), fill_value, dtype=image.dtype)
    else:
        crop = np.full((side, side), fill_value, dtype=image.dtype)

    src_x0 = max(0, crop_x0)
    src_y0 = max(0, crop_y0)
    src_x1 = min(w, crop_x1)
    src_y1 = min(h, crop_y1)

    dst_x0 = src_x0 - crop_x0
    dst_y0 = src_y0 - crop_y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    if src_x1 > src_x0 and src_y1 > src_y0:
        crop[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]

    return crop, crop_x0, crop_y0
