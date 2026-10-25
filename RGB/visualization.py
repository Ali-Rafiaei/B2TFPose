"""
visualization.py
Debug-visualization functions
"""

import os

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

from image_patch_utils import crop_bbox_square_centered, rotate_image


def visualize_high_level_nearest_neighbors(
    query_rgb,
    high_level_query_mask,
    top_views_closest_to_query,
    top_views_distances,
    input_to_encoder,
    templates_dir,
    obj_id,
    num_inplane_variations,
    inplane_rotation_angle,
    high_level_bg_weight: float = 1.0,
):
    if high_level_bg_weight > 0.0:
        bg = high_level_query_mask == 0
        third_image = query_rgb.astype(np.float32)
        third_image[bg] *= 0
        third_image = third_image.clip(0, 255).astype(np.uint8)
    else:
        third_image = query_rgb.astype(np.float32).clip(0, 255).astype(np.uint8)

    cropped_third = crop_bbox_square_centered(
        third_image, high_level_query_mask, return_mask=False
    )

    num_rows = 3
    num_cols = len(top_views_closest_to_query) // num_rows + 3
    plt.switch_backend("TkAgg")
    plt.figure(figsize=(20, 10))

    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(query_rgb)
    plt.title("Query Image", fontsize=16)
    plt.axis("off")

    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(input_to_encoder)
    plt.title("Input to Encoder", fontsize=16)
    plt.axis("off")

    plt.subplot(num_rows, num_cols, 3)
    plt.imshow(cropped_third)
    plt.title(
        "Segmented Q Image" if high_level_bg_weight > 0 else "Cropped Q Image",
        fontsize=16,
    )
    plt.axis("off")

    for j in range(len(top_views_closest_to_query)):
        template_view_data_idx = top_views_closest_to_query[j]
        template_base_view_id = template_view_data_idx // num_inplane_variations
        template_inplane_rot = (
            template_view_data_idx % num_inplane_variations * inplane_rotation_angle
        )
        template_inplane_rot *= -1

        template_base_view_path = os.path.join(
            f"{templates_dir}",
            f"{obj_id:06d}",
            (
                "train_pbr/000000/rgb"
                if "new_templates" in templates_dir
                else "rgb"
            ),
            f"{template_base_view_id:06d}.png",
        )
        template_base_view_mask_path = os.path.join(
            f"{templates_dir}",
            f"{obj_id:06d}",
            (
                "train_pbr/000000/mask_visib"
                if "new_templates" in templates_dir
                else "mask_visib"
            ),
            f"{template_base_view_id:06d}_000000.png",
        )
        template_base_view = np.array(Image.open(template_base_view_path))[:, :, :3]
        template_view = rotate_image(template_base_view, template_inplane_rot)
        template_base_view_mask = np.array(Image.open(template_base_view_mask_path))
        template_base_view_mask[template_base_view_mask > 0] = 1
        template_view_mask = rotate_image(
            template_base_view_mask, template_inplane_rot
        )
        segmented_template_view = (
            template_view * template_view_mask[:, :, np.newaxis]
        )
        input_to_encoder_view = crop_bbox_square_centered(
            segmented_template_view, template_view_mask
        )
        plt.subplot(num_rows, num_cols, j + 4)
        plt.imshow(input_to_encoder_view)
        plt.title(f"Dist {top_views_distances[j]:06f}", fontsize=16)
        plt.axis("off")

    plt.show()
    plt.close()


def visualize_initial_pnp_poses(
    initial_pnp_pose,
    rrc_pose,
    query_K,
    hypothesis_refiner,
):
    plt.switch_backend("tkagg")
    plt.figure(figsize=(20, 10))
    for k, (p, title) in enumerate(
        [
            (initial_pnp_pose, "initial PnP"),
            (rrc_pose, "rend-recorr PnP"),
        ]
    ):
        rgb_render, _, _ = hypothesis_refiner.render_pose_rgb(
            p, query_K, width=640, height=480
        )
        plt.subplot(1, 2, k + 1)
        plt.imshow(rgb_render)
        plt.title(title, fontsize=16)
        plt.axis("off")
    plt.suptitle("Estimated Poses of Initial RANSAC+PnP", fontsize=24)
    plt.show()
    plt.close()


def visualize_megapose_pose(
    final_pose,
    query_K,
    hypothesis_refiner,
):
    plt.switch_backend("tkagg")
    plt.figure(figsize=(10, 10))
    rgb_render, _, _ = hypothesis_refiner.render_pose_rgb(
        final_pose, query_K, width=640, height=480
    )
    plt.imshow(rgb_render)
    plt.title("MegaPose refined", fontsize=16)
    plt.axis("off")
    plt.suptitle("Estimated Pose after MegaPose Refinement", fontsize=24)
    plt.show()
    plt.close()


def visualize_final_icp_refinement(
    high_level_query_rgb,
    high_level_query_mask,
    query_K,
    poses_list,
    labels_list,
    hypothesis_refiner,
):
    H, W, _ = high_level_query_rgb.shape
    n_poses = len(poses_list)
    n_cols = n_poses + 1

    plt.switch_backend("TkAgg")
    _, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    axes[0, 0].imshow(high_level_query_rgb)
    axes[0, 0].set_title("Test Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(high_level_query_rgb)
    axes[1, 0].set_title("Query (ref)")
    axes[1, 0].axis("off")

    dark_red = np.array([139, 0, 0], dtype=np.uint8)

    for i, (pose, label) in enumerate(zip(poses_list, labels_list)):
        rgb, mask, _ = hypothesis_refiner.render_pose_rgb(pose, query_K, width=W, height=H)
        axes[0, i + 1].imshow(rgb)
        axes[0, i + 1].set_title(f"{label}")
        axes[0, i + 1].axis("off")
        overlay = high_level_query_rgb.copy()
        overlay[mask > 0] = dark_red
        axes[1, i + 1].imshow(overlay)
        axes[1, i + 1].set_title(f"{label} (overlay)")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
    input("Press Enter to continue...")
