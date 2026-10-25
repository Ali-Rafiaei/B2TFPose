"""Builds the shared per-object pipeline components (DINO backbone, template
cache, correspondence/hypothesis/mask pipeline, TeaserPP/MegaPose refiners)
used by both bop_inference.py and live_demo_inference.py."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
from megapose.refiner.run_inference_on_example import MegaPoseRefiner

from correspondence import CorrespondenceFinder
from dino_backbone import load_dino_model
from mask_pipeline import MaskPipeline
from refinement import HypothesisRefiner
from teaserpp_refiner import TeaserppRefiner
from template_cache import TemplateCache

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_POSES_PATH = os.path.join(
    _REPO_ROOT,
    "predefined_template_poses/obj_poses_level2_icosphere.npy",
)


@dataclass
class PipelineComponents:
    obj_model: o3d.geometry.TriangleMesh
    storage_full_path: str
    template_cache: TemplateCache
    hypothesis_refiner: HypothesisRefiner
    depth_refiner_type: str
    mask_pipeline: MaskPipeline
    teaserpp_refiner: Optional[TeaserppRefiner] = None
    megapose_refiner: Optional[MegaPoseRefiner] = None


def build_pipeline_components(
    configs,
    obj_id,
    obj_model_path,
    dataset_name,
    obj_diameter: Optional[float] = None,
    model_scale_to_mm: float = 1.0,
) -> PipelineComponents:
    # ── DINO backbone ────────────────────────────────────────────────────────
    dino_model_name = configs.get("dino_highlevel_model", "dinov3_vitl16")
    print(f"Loading {dino_model_name} for encoding...")
    (
        dino_model,
        dino_patch_size,
        dino_preprocess,
        dino_input_size,
        encoding_grid_size,
    ) = load_dino_model(configs)
    print(f"{dino_model_name} loaded successfully!")
    print(
        f"[Model] patch_size={dino_patch_size} (read from model.patch_embed.patch_size)"
    )

    hl_resize = round(dino_input_size * 256 / 224)
    print(
        f"[Model] input size: {dino_input_size}×{dino_input_size} "
        f"(resize={hl_resize} → crop={dino_input_size}, "
        f"patch_size={dino_patch_size}, "
        f"tokens={dino_input_size//dino_patch_size}²={(dino_input_size//dino_patch_size)**2})"
    )
    use_intermediate = configs.get("dino_patch_use_intermediate_layer", False)
    intermediate_layer_idx = configs.get("dino_patch_intermediate_layer_idx", None)
    dino_patch_layers = configs.get("dino_patch_layers", None)
    if dino_patch_layers is not None:
        print(
            f"[PATCH-LEVEL DINO] Multi-layer mode: concatenating layers {dino_patch_layers} "
            f"(feature dim = {len(dino_patch_layers)} × D)"
        )
    elif use_intermediate:
        print(
            f"[PATCH-LEVEL DINO] Using intermediate features of DINO "
            f"from layer with index {intermediate_layer_idx} for patch-level encoding"
        )
    else:
        print(
            "[PATCH-LEVEL DINO] Using final output features of DINO for patch-level encoding"
        )

    inplane_rotation_angle = configs["inplane_rotation_angle"]
    num_inplane_variations = 360 // inplane_rotation_angle

    # ── Template poses & camera intrinsics ──────────────────────────────────
    list_of_template_transformations = np.load(TEMPLATE_POSES_PATH)
    if "gigapose_templates" in configs.get("templates_dir", ""):
        print(
            "[INFO] Overriding template translations to (0, 0, 0.4) for gigapose templates..."
        )
        list_of_template_transformations[:, :3, 3] = np.array([0.0, 0.0, 0.4])
    # Convert template translations to mm.
    list_of_template_transformations[:, :3, 3] *= 1000

    # Pre-compute inverse poses once (avoids np.linalg.inv inside correspondence loop).
    inv_template_transformations = np.linalg.inv(list_of_template_transformations)

    template_camera_data_path = os.path.join(
        configs["templates_dir"], f"{obj_id:06d}", "camera.json"
    )
    with open(template_camera_data_path, "r") as f:
        template_camera_data = json.load(f)
    fx, fy = template_camera_data["fx"], template_camera_data["fy"]
    cx, cy = template_camera_data["cx"], template_camera_data["cy"]
    template_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # ── Object mesh, geo-IoU points, RANSAC threshold ───────────────────────
    obj_model = o3d.io.read_triangle_mesh(obj_model_path)

    # Pre-sample CAD model surface points for geometric IoU scoring.
    model_verts = np.asarray(obj_model.vertices, dtype=np.float32) * model_scale_to_mm
    if len(model_verts) > 2000:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(model_verts), 2000, replace=False)
        model_verts = model_verts[idx]
    # (N≤2000, 3) float32, mm.
    geo_iou_model_pts = model_verts

    if obj_diameter is None:
        # No override available — estimate from mesh vertices.
        all_verts = np.asarray(obj_model.vertices, dtype=np.float64) * model_scale_to_mm
        c = all_verts.mean(axis=0)
        obj_diameter = float(np.max(np.linalg.norm(all_verts - c, axis=1)) * 2.0)
        print(
            f"[Init] Computed obj_diameter = {obj_diameter:.1f} mm from model vertices."
        )

    thresh_opt = configs["ransac_inlier_threshold"]
    if isinstance(thresh_opt, float):
        ransac_threshold = thresh_opt
    elif isinstance(thresh_opt, str) and thresh_opt == "dynamic":
        # 8% of object diameter (matches the best static value found for YCBV).
        ransac_threshold = 0.08 * obj_diameter

    # D11 fix: this used to read "dino_highlevel_intermediate_layer_idx", a key
    # nothing ever sets (it's commented out in both default configs). The
    # global/high-level descriptor layer actually used (correspondence.py,
    # template_cache.py) is dino_global_layer_idx, which genuinely differs
    # across datasets (18 for lmo, 23 for tudl) - two configs that only differ
    # by that value were silently sharing one cache dir.
    hl_layer = configs.get("dino_global_layer_idx", None)
    hl_layer_tag = f"hlLayer{hl_layer}" if hl_layer is not None else "hlLayerLast"
    p_layer_tag = f"patchLayer{intermediate_layer_idx}"
    storage_full_path = os.path.join(
        "data",
        dataset_name,
        dino_model_name,
        f"hlIn{dino_input_size}_patchIn{dino_input_size}_{hl_layer_tag}_{p_layer_tag}",
        f"embeddings_and_labels_{str(obj_id).zfill(6)}",
    )
    if not os.path.exists(storage_full_path):
        os.makedirs(storage_full_path)

    # ── Component construction ──────────────────────────────────────────────
    template_cache = TemplateCache(
        configs=configs,
        obj_id=obj_id,
        dino_model=dino_model,
        dino_input_size=dino_input_size,
        storage_full_path=storage_full_path,
        num_inplane_variations=num_inplane_variations,
        inplane_rotation_angle=inplane_rotation_angle,
    )

    correspondence_finder = CorrespondenceFinder(
        configs=configs,
        dino_model=dino_model,
        dino_preprocess=dino_preprocess,
        dino_input_size=dino_input_size,
        num_inplane_variations=num_inplane_variations,
        inplane_rotation_angle=inplane_rotation_angle,
        list_of_template_transformations=list_of_template_transformations,
        inv_template_transformations=inv_template_transformations,
        template_intrinsics=template_intrinsics,
        geo_iou_model_pts=geo_iou_model_pts,
        template_cache=template_cache,
    )

    hypothesis_refiner = HypothesisRefiner(
        configs=configs,
        obj_model_path=obj_model_path,
        obj_diameter=obj_diameter,
        correspondence_finder=correspondence_finder,
    )

    depth_refiner_type = configs.get("depth_refiner", "icp")
    teaserpp_refiner = None
    if depth_refiner_type == "teaserpp":
        print("Initializing Teaser++ refiner...")
        teaserpp_refiner = TeaserppRefiner(
            mesh_path=obj_model_path,
            noise_bound=configs.get("teaserpp_noise_bound", 0.015),
            n_points=configs.get("teaserpp_n_points", 500),
            n_min_points=configs.get("teaserpp_n_min_points", 50),
            min_num_inliers=configs.get("teaserpp_min_inliers", 30),
            use_farthest_point_sampling=configs.get("teaserpp_use_fps", False),
        )
        print("Teaser++ refiner initialized!")

    megapose_mode = None
    if "rgbd_megapose" in configs.get("refinements", []):
        megapose_mode = "rgbd"
    elif "rgb_megapose" in configs.get("refinements", []):
        megapose_mode = "rgb"

    megapose_refiner = None
    if megapose_mode is not None:
        print(f"Initializing MegaPose refiner (mode={megapose_mode})...")
        megapose_refiner = MegaPoseRefiner(obj_model_path, mode=megapose_mode)
        print("MegaPose refiner initialized!")

    mask_pipeline = MaskPipeline(
        configs=configs,
        correspondence_finder=correspondence_finder,
        hypothesis_refiner=hypothesis_refiner,
        storage_full_path=storage_full_path,
        obj_model_path=obj_model_path,
        ransac_threshold=ransac_threshold,
        depth_refiner_type=depth_refiner_type,
        num_inplane_variations=num_inplane_variations,
        inplane_rotation_angle=inplane_rotation_angle,
        obj_id=obj_id,
        megapose_refiner=megapose_refiner,
        megapose_mode=megapose_mode,
    )

    return PipelineComponents(
        obj_model=obj_model,
        storage_full_path=storage_full_path,
        template_cache=template_cache,
        hypothesis_refiner=hypothesis_refiner,
        depth_refiner_type=depth_refiner_type,
        mask_pipeline=mask_pipeline,
        teaserpp_refiner=teaserpp_refiner,
        megapose_refiner=megapose_refiner,
    )
