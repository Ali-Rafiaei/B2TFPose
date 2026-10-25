"""DINO backbone loading, shared by bop_inference.py and live_demo_inference.py.
A single model is used for both high-level (global) and patch-level (dense)
encoding."""

import os
from glob import glob

import torch
from torchvision import transforms


def build_dino_preprocess(input_size):
    """Standard DINO preprocessing: resize, center-crop to input_size, ImageNet-normalize."""
    resize_size = round(input_size * 256 / 224)
    preprocessor = transforms.Compose(
        [
            transforms.Resize(
                resize_size, interpolation=transforms.InterpolationMode.BICUBIC
            ),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocessor


def load_dino_model(configs: dict):
    """Loads the DINO backbone — one model shared for both high-level (global)
    and patch-level (dense) encoding, via different layers/pooling on the same
    forward pass. Returns (dino_model, patch_size, preprocess_fn, input_size,
    encoding_grid_size)."""
    dino_model_name = configs.get("dino_highlevel_model", "dinov3_vitl16")

    if dino_model_name.startswith("dinov3_"):
        dinov3_repo_dir = configs["dinov3_repo_dir"]

        weights_dir = os.path.join(dinov3_repo_dir, "weights")
        candidates = glob(os.path.join(weights_dir, f"{dino_model_name}*"))
        dinov3_weights_path = candidates[0]

        dino_model = torch.hub.load(
            dinov3_repo_dir,
            dino_model_name,
            source="local",
            weights=dinov3_weights_path,
        )
    else:
        dino_model = torch.hub.load("facebookresearch/dinov2", dino_model_name)

    dino_model.eval().cuda()

    # Read patch size directly from the loaded model.
    try:
        ps = dino_model.patch_embed.patch_size
        dino_patch_size = ps[0] if isinstance(ps, (tuple, list)) else int(ps)
    except AttributeError:
        dino_patch_size = 16 if dino_model_name.startswith("dinov3_") else 14

    encoding_grid_size = configs.get("encoding_grid_size", 32)
    dino_input_size = dino_patch_size * encoding_grid_size

    dino_preprocess = build_dino_preprocess(dino_input_size)

    return (
        dino_model,
        dino_patch_size,
        dino_preprocess,
        dino_input_size,
        encoding_grid_size,
    )
