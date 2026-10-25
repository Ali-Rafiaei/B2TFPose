"""Builds the shared per-object pipeline components (DINO backbone, template
cache, correspondence/refinement/mask pipeline, MegaPose refiner) used by
bop_inference.py."""

import json
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from megapose.refiner.run_inference_on_example import MegaPoseRefiner

from correspondence import CorrespondenceFinder
from dino_backbone import load_dino_model
from mask_pipeline import MaskPipeline
from refinement import HypothesisRefiner
from template_cache import TemplateCache

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATE_POSES_PATH = os.path.join(
    _REPO_ROOT,
    "predefined_template_poses/obj_poses_level2_icosphere.npy",
)


@dataclass
class PipelineComponents:
    storage_full_path: str
    template_cache: TemplateCache
    correspondence_finder: CorrespondenceFinder
    hypothesis_refiner: HypothesisRefiner
    mask_pipeline: MaskPipeline
    megapose_refiner: Optional[MegaPoseRefiner] = None
    megapose_mode: Optional[str] = None


def build_pipeline_components(
    configs,
    obj_id,
    obj_model_path,
    dataset_name,
) -> PipelineComponents:
    # ── DINO backbone ────────────────────────────────────────────────────────
    dino_model_name = configs.get("dino_highlevel_model", "dinov3_vitl16")
    (
        dino_model,
        dino_patch_size,
        dino_preprocess,
        dino_input_size,
        _encoding_grid_size,
    ) = load_dino_model(configs)

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
    list_of_template_transformations[:, :3, 3] *= 1000  # to mm

    inv_template_transformations = np.linalg.inv(list_of_template_transformations)

    template_camera_data_path = os.path.join(
        configs["templates_dir"], f"{obj_id:06d}", "camera.json"
    )
    with open(template_camera_data_path, "r") as f:
        template_camera_data = json.load(f)
    fx, fy = template_camera_data["fx"], template_camera_data["fy"]
    cx, cy = template_camera_data["cx"], template_camera_data["cy"]
    template_intrinsics = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])

    # ── Storage path ─────────────────────────────────────────────────────────
    # D11 fix: this used to read "dino_highlevel_intermediate_layer_idx", a key
    # nothing ever sets. The global/high-level descriptor layer actually used
    # (correspondence.py, template_cache.py) is cache_v2_global_layer_idx, so two
    # configs that only differ by that value were silently sharing one cache dir.
    hl_layer = configs.get("cache_v2_global_layer_idx", None)
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

    # ── MegaPose refiner ─────────────────────────────────────────────────────
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
        template_cache=template_cache,
    )

    hypothesis_refiner = HypothesisRefiner(
        configs=configs,
        obj_model_path=obj_model_path,
        dataset_name=dataset_name,
        correspondence_finder=correspondence_finder,
    )

    mask_pipeline = MaskPipeline(
        configs=configs,
        correspondence_finder=correspondence_finder,
        hypothesis_refiner=hypothesis_refiner,
        storage_full_path=storage_full_path,
        obj_id=obj_id,
        num_inplane_variations=num_inplane_variations,
        inplane_rotation_angle=inplane_rotation_angle,
        megapose_refiner=megapose_refiner,
        megapose_mode=megapose_mode,
    )

    return PipelineComponents(
        storage_full_path=storage_full_path,
        template_cache=template_cache,
        correspondence_finder=correspondence_finder,
        hypothesis_refiner=hypothesis_refiner,
        mask_pipeline=mask_pipeline,
        megapose_refiner=megapose_refiner,
        megapose_mode=megapose_mode,
    )
