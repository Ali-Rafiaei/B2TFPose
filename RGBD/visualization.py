"""
visualization.py
Debug-visualization functions
"""

import os

import numpy as np
import open3d as o3d
from matplotlib import pyplot as plt
from PIL import Image

from image_patch_utils import crop_bbox_square_centered, rotate_image


def visualize_high_level_nearest_neighbors(
    query_rgb,
    top_views_closest_to_query,
    top_views_distances,
    input_to_encoder,
    templates_dir,
    obj_id,
    num_inplane_variations,
    inplane_rotation_angle,
):
    num_rows = 3
    num_cols = len(top_views_closest_to_query) // num_rows + 2
    plt.switch_backend("TkAgg")
    plt.figure(figsize=(20, 10))
    plt.subplot(num_rows, num_cols, 1)
    plt.imshow(query_rgb)
    plt.title(f"Query Image", fontsize=16)
    plt.axis("off")
    plt.subplot(num_rows, num_cols, 2)
    plt.imshow(input_to_encoder)
    plt.title(f"Input to Encoder", fontsize=16)
    plt.axis("off")

    for j in range(len(top_views_closest_to_query)):
        template_view_data_idx = top_views_closest_to_query[j]
        template_base_view_id = template_view_data_idx // num_inplane_variations
        template_inplane_rot = (
            template_view_data_idx % num_inplane_variations * inplane_rotation_angle
        )
        # Negate the rotation for visualization.
        template_inplane_rot *= -1

        template_base_view_path = os.path.join(
            f"{templates_dir}",
            f"{obj_id:06d}",
            "train_pbr/000000/rgb" if "new_templates" in templates_dir else "rgb",
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
        template_view_mask = rotate_image(template_base_view_mask, template_inplane_rot)
        segmented_template_view = template_view * template_view_mask[:, :, np.newaxis]

        input_to_encoder_view = crop_bbox_square_centered(
            segmented_template_view, template_view_mask
        )

        plt.subplot(num_rows, num_cols, j + 3)
        plt.imshow(input_to_encoder_view)
        plt.title(f"Dist {top_views_distances[j]:06f}", fontsize=16)
        plt.axis("off")

    plt.show()
    plt.close()


def visualize_pose_alignment_3d(
    poses_list: list,
    labels_list: list,
    model_points: np.ndarray,
    query_scene_points: np.ndarray,
):
    """Opens one blocking Open3D window per pose: query points (green) vs.
    the model transformed by that pose (red). Close a window to advance."""
    scene_pcd = o3d.geometry.PointCloud()
    scene_pcd.points = o3d.utility.Vector3dVector(query_scene_points)
    scene_pcd.paint_uniform_color([0.0, 0.8, 0.0])

    for pose, label in zip(poses_list, labels_list):
        R = pose[:3, :3]
        t = pose[:3, 3]
        transformed = (R @ model_points.T + t[:, np.newaxis]).T

        model_pcd = o3d.geometry.PointCloud()
        model_pcd.points = o3d.utility.Vector3dVector(transformed)
        model_pcd.paint_uniform_color([0.9, 0.1, 0.1])

        o3d.visualization.draw_geometries(
            [scene_pcd, model_pcd],
            window_name=f"{label}  |  close to continue",
            width=1000,
            height=700,
        )


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
    # +1 for the query image column.
    n_cols = n_poses + 1

    plt.switch_backend("TkAgg")
    fig, axes = plt.subplots(2, n_cols, figsize=(5 * n_cols, 10))
    # Ensure axes is always 2D
    if n_cols == 1:
        axes = axes.reshape(2, 1)

    # Column 0: query image (top) and a reference copy (bottom)
    axes[0, 0].imshow(high_level_query_rgb)
    axes[0, 0].set_title("Test Image")
    axes[0, 0].axis("off")
    axes[1, 0].imshow(high_level_query_rgb)
    axes[1, 0].set_title("Query (ref)")
    axes[1, 0].axis("off")

    dark_red = np.array([139, 0, 0], dtype=np.uint8)

    for i, (pose, label) in enumerate(zip(poses_list, labels_list)):
        rgb, mask, _ = hypothesis_refiner.render_pose_rgb(
            pose,
            query_K,
            width=W,
            height=H,
        )
        # Top row: rendered pose
        axes[0, i + 1].imshow(rgb)
        axes[0, i + 1].set_title(f"{label}")
        axes[0, i + 1].axis("off")

        # Bottom row: query image with rendered object region painted dark red
        overlay = high_level_query_rgb.copy()
        overlay[mask > 0] = dark_red
        axes[1, i + 1].imshow(overlay)
        axes[1, i + 1].set_title(f"{label} (overlay)")
        axes[1, i + 1].axis("off")

    plt.tight_layout()
    plt.show()
    plt.close()
    input("Press Enter to continue...")


def visualize_pose_hypotheses_grid(
    poses_list,
    query_K,
    width,
    height,
    hypothesis_refiner,
    title,
):
    """Renders poses_list in a 3x5 grid. Use the actual query image's
    width/height, not a fixed resolution, so aspect ratio matches the scene."""
    plt.switch_backend("tkagg")
    plt.figure(figsize=(40, 20))
    for i, pose in enumerate(poses_list):
        rgb_render, _, _ = hypothesis_refiner.render_pose_rgb(
            pose,
            query_K,
            width=width,
            height=height,
        )
        plt.subplot(3, 5, i + 1)
        plt.imshow(rgb_render)
        plt.axis("off")
    plt.suptitle(title, fontsize=18)
    plt.show()
    plt.close()
