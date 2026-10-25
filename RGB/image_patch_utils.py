import cv2
import numpy as np
from typing import Union


def extract_crop_with_coords(
    image: np.ndarray,
    center_x: float,
    center_y: float,
    side: int,
    fill_value: int = 0,
) -> tuple:
    """Crop a fixed-size square centered at (center_x, center_y), zero-padded
    at the edges. Returns (crop, crop_x0, crop_y0) — crop_x0/y0 are the
    original-image coordinates of the crop's top-left corner, needed to map
    crop-local pixel coordinates back to the original image."""
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


def rotate_image(image: np.ndarray, angle: float):
    center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_image = cv2.warpAffine(
        image, rot_mat, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR
    )

    return rotated_image
