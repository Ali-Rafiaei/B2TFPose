"""
Copyright (c) 2022 Inria & NVIDIA CORPORATION & AFFILIATES. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""


# Standard Library
import time
from typing import Optional, Tuple

# Third Party
import numpy as np
import numpy.typing as npt
import teaserpp_python
import torch
import pyrender
import trimesh
import open3d as o3d


def get_pointcloud(depth, intrinsics, flatten=False, remove_zero_depth_points=True):
    """Backprojects an HxW depth map (meters) to an HxWx3 (or flattened) pointcloud."""

    # height, width = depth.shape
    # xlin = np.linspace(0, width - 1, width)
    # ylin = np.linspace(0, height - 1, height)
    # px, py = np.meshgrid(xlin, ylin)
    # px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
    # py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
    # points = np.float32([px, py, depth]).transpose(1, 2, 0)
    #
    # # if remove_zero_depth_points:
    # #     mask = depth > 0
    # #     points = points[mask]
    #
    # if flatten:
    #     points = np.reshape(points, [height * width, 3])

    # Above code works but is slow, so using more optimized version below that gets rid of zeros and outputs flattened pointcloud directly
    u, v = np.nonzero(depth)
    z = depth[u, v]
    x = (v - intrinsics[0, 2]) * z / intrinsics[0, 0]
    y = (u - intrinsics[1, 2]) * z / intrinsics[1, 1]
    points = np.stack((x, y, z), axis=-1).astype(np.float32)

    return points

def render_mesh_at_pose(mesh, pose_4x4, K, resolution):

    height, width = resolution

    render_pose_adapter = np.eye(4, dtype=np.float32)
    render_pose_adapter[1, 1] = -1.0
    render_pose_adapter[2, 2] = -1.0

    pose_gl = (render_pose_adapter @ pose_4x4).astype(np.float32)
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        ambient_light=np.array([0.03, 0.03, 0.03, 1.0], dtype=np.float32),
    )
    scene.add(mesh, pose=pose_gl)

    cam = pyrender.IntrinsicsCamera(
        fx=float(K[0, 0]),
        fy=float(K[1, 1]),
        cx=float(K[0, 2]),
        cy=float(K[1, 2]),
        znear=10,
        zfar=2000.0,
    )
    scene.add(cam, pose=np.eye(4, dtype=np.float32))

    light_pose = np.eye(4, dtype=np.float32)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=25.5), pose=light_pose
    )
    scene.add(pyrender.PointLight(color=np.ones(3), intensity=5.0), pose=light_pose)

    # # Warm ceiling light above the camera looking straight down
    # ceiling_pose = np.eye(4, dtype=np.float32)
    # ceiling_pose[:3, 3] = [0.0, 0.0, 1.5]   # 1.5 m above origin; tweak as needed
    # ceiling_color = np.array([1.0, 0.55, 0.25])  # orange-ish
    # scene.add(
    #   pyrender.SpotLight(color=ceiling_color, intensity=250.0,
    #                      innerConeAngle=np.deg2rad(25),
    #                      outerConeAngle=np.deg2rad(40)),
    #   pose=ceiling_pose,
    # )

    renderer = pyrender.OffscreenRenderer(viewport_width=width, viewport_height=height)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    mask = (depth > 0).astype(np.uint8)
    rgb = color[..., :3]
    return rgb, depth, mask

def numpy_to_open3d(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    return pcd

def compute_depth_consistency_masks(mask_type, depth_rendered, depth_measured, depth_delta_thresh=0.1):
    """Masks valid pixels for depth-consistency checks: "simple" keeps any
    rendered+measured overlap, "threshold" also requires the depth values to
    agree within depth_delta_thresh."""

    mask_rendered = depth_rendered > 0
    mask_measured = np.logical_and(depth_measured > 0, depth_rendered > 0)

    if mask_type == "simple":
        pass
    elif mask_type == "threshold":
        depth_delta = np.abs(depth_measured - depth_rendered)
        mask_measured[depth_delta > depth_delta_thresh] = 0
    else:
        raise ValueError(f"Unknown mask type {mask_type}")

    # Most conservative
    mask_rendered = mask_measured

    return mask_rendered, mask_measured

def get_solver_params(noise_bound: float = 0.01):
    solver_params = teaserpp_python.RobustRegistrationSolver.Params()
    solver_params.cbar2 = 1
    solver_params.noise_bound = noise_bound
    solver_params.estimate_scaling = False
    solver_params.rotation_estimation_algorithm = (
        teaserpp_python.RobustRegistrationSolver.ROTATION_ESTIMATION_ALGORITHM.GNC_TLS
    )
    solver_params.rotation_gnc_factor = 1.4
    solver_params.rotation_max_iterations = 100
    solver_params.rotation_cost_threshold = 1e-12

    return solver_params


def compute_teaserpp_refinement(
    depth_src: npt.ArrayLike,
    depth_tgt: npt.ArrayLike,
    mask_src: npt.ArrayLike,
    mask_tgt: npt.ArrayLike,
    cam_K: npt.ArrayLike,
    solver_params=None,
    max_num_points=None,
    normals_src=None,
    use_farthest_point_sampling: bool = True,
    **solver_params_kwargs
) -> dict:
    """Teaser++ registration (see the MIT-SPARK/TEASER-plusplus minimal
    Python example). Returns a dict with 'T_tgt_src', the rigid transform
    that aligns src to tgt."""

    if solver_params is None:
        solver_params = get_solver_params(**solver_params_kwargs)

    depth_src_masked = depth_src.copy()
    depth_src_masked[mask_src == 0] = 0
    depth_tgt_masked = depth_tgt.copy()
    depth_tgt_masked[mask_tgt == 0] = 0

    pc_src_masked = get_pointcloud(depth_src_masked, cam_K)
    pc_tgt_masked = get_pointcloud(depth_tgt_masked, cam_K)

    if max_num_points is not None:
        N = pc_src_masked.shape[0]
        M = min(max_num_points, N)

        if normals_src is None:
            x = torch.tensor(pc_src_masked).cuda()
        else:
            a = torch.tensor(pc_src_masked).cuda()
            b = torch.tensor(normals_src[mask_src]).cuda()
            x = torch.cat((a, b), dim=-1)

        # Do furthest point sampling with pytorch3d
        # [1, N, 3]
        x = x.unsqueeze(0)
        lengths = torch.tensor([x.shape[1]]).cuda()
        K = max_num_points

        idx = None
        fps_succeeded = False
        if use_farthest_point_sampling:
            try:
                import pytorch3d.ops
                _, idx = pytorch3d.ops.sample_farthest_points(x, lengths, K)
                idx = idx.squeeze().cpu().numpy()
                fps_succeeded = True
            except (ImportError, Exception):
                print("[Teaser++] pytorch3d FPS unavailable, falling back to random sampling")

        if not fps_succeeded:
            idx = np.random.choice(
                np.arange(pc_src_masked.shape[0]),
                size=min(max_num_points, pc_src_masked.shape[0]),
                replace=False,
            )

        pc_src = pc_src_masked[idx].astype(np.float64)
        pc_tgt = pc_tgt_masked[idx].astype(np.float64)


        # pc_src = pc_src_masked[idx].astype(np.float64)
        # pc_tgt = pc_tgt_masked[idx].astype(np.float64)
    else:
        pc_src = pc_src_masked.astype(np.float64)
        pc_tgt = pc_tgt_masked.astype(np.float64)

    solver = teaserpp_python.RobustRegistrationSolver(solver_params)
    start = time.time()

    # teaserpp wants [3,N] pointclouds
    solver.solve(pc_src.transpose(), pc_tgt.transpose())
    end = time.time()

    solution = solver.getSolution()

    T = np.eye(4)
    T[:3, :3] = solution.rotation
    T[:3, 3] = solution.translation
    T_tgt_src = T

    # check number of inliers
    pc_src_o3d = numpy_to_open3d(pc_src)
    pc_src_o3d_refined = pc_src_o3d.transform(T_tgt_src)
    pc_src_refined = np.array(pc_src_o3d_refined.points)

    diff = np.linalg.norm(pc_src_refined - pc_tgt, axis=-1)
    inliers = np.count_nonzero(diff < solver_params.noise_bound)

    return {
        "solution": solution,
        # "pc_src_raw": pc_src_raw,
        # "pc_tgt_raw": pc_tgt_raw,
        # "pc_src": pc_src,
        # "pc_tgt": pc_tgt,
        # "pc_src_mask": pc_src_mask,
        # "pc_tgt_mask": pc_tgt_mask,
        "T": T,
        # T_tgt_src is the transform that aligns src to tgt.
        "T_tgt_src": T,
        "num_inliers": inliers,
    }


class TeaserppRefiner():
    def __init__(
        self,
        mesh_path,
        mask_type: str = "simple",
        depth_delta_thresh: float = 0.1,
        n_min_points: int = 100,
        n_points: int = 1000,
        noise_bound: float = 0.01,
        min_num_inliers: int = 50,
        use_farthest_point_sampling: bool = False,
    ) -> None:
        self.mask_type = mask_type
        self.depth_delta_thresh = depth_delta_thresh
        self.n_min_points = n_min_points
        self.n_points = n_points
        self.noise_bound = noise_bound
        self.min_num_inliers = min_num_inliers
        self.use_farthest_point_sampling = use_farthest_point_sampling

        self.mesh_path = mesh_path
        mesh = trimesh.load(mesh_path, force="mesh")
        self.mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)


        self.debug = {}
        
    def refine_poses(
        self,
        prediction: np.ndarray,
        mask: np.ndarray,
        depth: np.ndarray,
        K: np.ndarray,
    ) -> np.ndarray:
        """Renders depth at the predicted pose, pairs it pixel-for-pixel with
        observed depth (assumes the pose is already aligned in image space),
        masks outliers, then refines with Teaser++."""
        # Detect mm vs. meters (BOP depth is mm, values >> 20) and scale the
        # meter-denominated noise_bound/depth_delta_thresh defaults to match.
        scale = 1.0
        if np.max(depth) > 20.0:
            scale = 1000.0

        prediction_refined = prediction.copy()
        resolution = depth.shape[-2:]

        # Apply the segmentation mask to measured depth so correspondences are
        # restricted to the object region (avoids background contamination).
        depth_segmented = depth.copy()
        if mask is not None:
            depth_segmented[mask == 0] = 0

        render_output = render_mesh_at_pose(
            self.mesh,
            prediction,
            K=K,
            resolution=resolution,
        )
        rendered_rgb, rendered_depth, _ = render_output

        # Validity mask: pixels where rendered and measured depth agree
        # (self.mask_type, default "simple"; threshold scaled to depth units).
        _, overlap_mask = compute_depth_consistency_masks(
            mask_type=self.mask_type,
            depth_rendered=rendered_depth,
            depth_measured=depth_segmented,
            depth_delta_thresh=self.depth_delta_thresh * scale,
        )

        if np.count_nonzero(overlap_mask) < self.n_min_points:
            print(
                f"[Teaser++] Insufficient overlap ({np.count_nonzero(overlap_mask)} "
                f"< {self.n_min_points}), returning original pose"
            )
            return prediction_refined

        out = compute_teaserpp_refinement(
            depth_src=rendered_depth,
            depth_tgt=depth_segmented,
            mask_src=overlap_mask,
            mask_tgt=overlap_mask,
            cam_K=K,
            max_num_points=self.n_points,
            noise_bound=self.noise_bound * scale,
            use_farthest_point_sampling=self.use_farthest_point_sampling,
        )

        # Only accept the refined pose when we have enough inlier support.
        if out["num_inliers"] >= self.min_num_inliers:
            T_tgt_src = out["T_tgt_src"]
            prediction_refined = T_tgt_src @ prediction_refined
            print(
                f"[Teaser++] Refinement applied — inliers: {out['num_inliers']}"
            )
        else:
            print(
                f"[Teaser++] Refinement skipped — insufficient inliers "
                f"({out['num_inliers']} < {self.min_num_inliers})"
            )

        self.debug = out
        return prediction_refined
