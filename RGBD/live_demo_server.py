#!/usr/bin/env python3
"""
live_demo_server.py
Flask server for B2TF RGBD live pose estimation demo.

Startup:
    python live_demo_server.py --config configs/demo_live.yaml

POST /estimate
    Input:  multipart/form-data
              rgb   — JPEG-compressed bytes of the scene RGB image
              depth — numpy bytes produced by np.save(buf, depth_float32_mm)
    Output: PNG overlay image (200 OK)  or  plain-text error (4xx / 5xx)
"""

import os
import sys

# Must be set before any pyrender / trimesh / OpenGL import.
os.environ["PYOPENGL_PLATFORM"] = "egl"

INFERENCE_DIR    = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT        = os.path.dirname(INFERENCE_DIR)
CNOS_DIR         = os.path.join(REPO_ROOT, "cnos")
CNOS_CONFIGS_DIR = os.path.join(CNOS_DIR, "configs")
# Makes `from src.model.utils import ...` resolve.
sys.path.insert(0, CNOS_DIR)

import io
import json
import logging
import tempfile
import time
from datetime import datetime

import cv2
import numpy as np
import torch
import trimesh
import pyrender
import matplotlib
matplotlib.use("Agg")
import glob
from flask import Flask, request, Response
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from live_demo_inference import LiveDemoPoseEstimator, parse_intrinsics
from config_utils import load_config_with_overrides

# CNOS imports — available after sys.path.insert(0, CNOS_DIR) above
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from src.model.utils import Detections, convert_npz_to_json
from src.model.loss import Similarity
from src.utils.bbox_utils import CropResizePad

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# Per-request files (frame.png, frame.npy, detections.json, overlay.png) are kept
# here for debugging instead of being deleted — inspect them after the demo if needed.
REQUESTS_DIR = os.path.join(INFERENCE_DIR, "live_demo_requests")
os.makedirs(REQUESTS_DIR, exist_ok=True)

# ── Overlay colors (9 visually distinct, low-red — object itself is red) ─────
COLORS_HEX = ["#00BFFF", "#00CED1", "#1E90FF", "#00FA9A", "#4B0082",
               "#008080", "#00FF7F", "#4169E1", "#2E8B57"]
COLORS_INT = [tuple(int(h.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
              for h in COLORS_HEX]

# ── Module-level state (populated during __main__ startup) ───────────────────
app = Flask(__name__)
_config: dict = {}
_K: np.ndarray = None
_b2tf: LiveDemoPoseEstimator = None
_cnos_model = None
# _cnos_ref_feats: GPU tensor, precomputed once at startup.
_cnos_ref_feats = None
# _cnos_metric: Similarity() instance, stateless, reused per request.
_cnos_metric = None
# _pyrender_mesh: pyrender.Mesh, loaded once at startup, reused per request.
_pyrender_mesh = None


# ── CNOS initialisation ───────────────────────────────────────────────────────

def init_cnos(config: dict) -> None:
    """Load FastSAM + DINOv2 models and precompute template reference features."""
    global _cnos_model, _cnos_ref_feats, _cnos_metric

    # Hydra allows only one initialize() per process; clear any prior state first.
    GlobalHydra.instance().clear()
    initialize_config_dir(config_dir=CNOS_CONFIGS_DIR, version_base=None)
    cfg = compose(config_name="run_inference.yaml")

    _cnos_metric = Similarity()

    # CNOS's Hydra config uses paths relative to cnos/, so CWD must be there
    # while instantiate() opens the SAM/DINOv2 checkpoint files.
    _orig_dir = os.getcwd()
    os.chdir(CNOS_DIR)
    try:
        model = instantiate(cfg.model)
    finally:
        os.chdir(_orig_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.descriptor_model.model = model.descriptor_model.model.to(device)
    model.descriptor_model.model.device = device
    if hasattr(model.segmentor_model, "predictor"):
        model.segmentor_model.predictor.model = (
            model.segmentor_model.predictor.model.to(device)
        )
    else:
        model.segmentor_model.model.setup_model(device=device, verbose=True)

    # Precompute DINOv2 reference features for all template images.
    template_dir = config["cnos_template_dir"]
    template_paths = sorted(glob.glob(os.path.join(template_dir, "*.png")))
    if not template_paths:
        raise FileNotFoundError(f"No PNG templates found in: {template_dir}")

    boxes, templates = [], []
    for path in template_paths:
        img = Image.open(path)
        boxes.append(img.getbbox())
        img_tensor = torch.from_numpy(np.array(img.convert("RGB")) / 255.0).float()
        templates.append(img_tensor)

    templates = torch.stack(templates).permute(0, 3, 1, 2)
    boxes = torch.tensor(np.array(boxes))
    processing_config = OmegaConf.create({"image_size": 224})
    processor = CropResizePad(processing_config.image_size)
    templates_processed = processor(images=templates, boxes=boxes).cuda()

    _cnos_ref_feats = model.descriptor_model.compute_features(
        templates_processed, token_name="x_norm_clstoken"
    )
    _cnos_model = model
    log.info(f"[CNOS] Ready — {len(template_paths)} templates, "
             f"ref_feats shape={list(_cnos_ref_feats.shape)}")


def run_cnos_segmentation(rgb_array: np.ndarray, num_max_dets: int,
                          conf_threshold: float) -> list:
    """Run FastSAM + DINOv2 scoring in-memory. Returns list of detection dicts."""
    detections = _cnos_model.segmentor_model.generate_masks(rgb_array)
    detections = Detections(detections)
    descriptors = _cnos_model.descriptor_model.forward(rgb_array, detections)

    # Score each proposal against the reference templates.
    scores = _cnos_metric(descriptors[:, None, :], _cnos_ref_feats[None, :, :])
    k_feat = min(5, scores.shape[-1])
    score_per_detection = torch.topk(scores, k=k_feat, dim=-1)[0]
    score_per_detection = torch.mean(score_per_detection, dim=-1)

    k = min(num_max_dets, score_per_detection.shape[0])
    scores, index = torch.topk(score_per_detection, k=k, dim=-1)
    detections.filter(index)
    detections.filter(scores > conf_threshold)
    detections.add_attribute("scores", scores)
    detections.add_attribute("object_ids", torch.zeros_like(scores))
    detections.to_numpy()

    # Convert to the JSON-serialisable list that run_demo expects.
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "detection")
        detections.save_to_file(0, 0, 0, save_path, "custom", return_results=False)
        result = convert_npz_to_json(idx=0, list_npz_paths=[save_path + ".npz"])
    return result


# ── Pose overlay renderer ─────────────────────────────────────────────────────

def render_overlay(rgb_array: np.ndarray, poses: list, K: np.ndarray,
                   mesh: pyrender.Mesh, annotation_text: str = None) -> bytes:
    """
    Render colored silhouettes for each estimated pose onto rgb_array.
    Returns PNG bytes.

    If `annotation_text` is given, it is drawn in the top-left corner on a solid
    background box (for legibility regardless of scene content) — used to display the
    pipeline-only timing measurement (see §8.2 in live_demo_doc.md: this is shown to the
    audience to demonstrate that round-trip latency is dominated by network transfer over
    Tailscale, not by the pose-estimation pipeline itself).
    """
    H, W = rgb_array.shape[:2]
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    # OpenGL convention: flip Y and Z relative to camera frame used in the pipeline.
    adapter = np.eye(4, dtype=np.float64)
    adapter[1, 1] = -1.0
    adapter[2, 2] = -1.0

    scn = pyrender.Scene(bg_color=[0, 0, 0, 0], ambient_light=[0.5, 0.5, 0.5, 1.0])
    mesh_node = scn.add(mesh, pose=np.eye(4, dtype=np.float32))
    cam = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=5000.0)
    scn.add(cam, pose=np.eye(4, dtype=np.float32))
    scn.add(pyrender.DirectionalLight(color=np.ones(3), intensity=5.5),
            pose=np.eye(4, dtype=np.float32))
    renderer = pyrender.OffscreenRenderer(W, H)

    overlay = rgb_array.astype(np.float32).copy()
    alpha = 0.6

    for i, entry in enumerate(poses):
        R = np.array(entry["R"], dtype=np.float64)
        t = np.array(entry["t_mm"], dtype=np.float64)
        pose_4x4 = np.eye(4)
        pose_4x4[:3, :3] = R
        pose_4x4[:3, 3] = t
        pose_gl = (adapter @ pose_4x4).astype(np.float32)
        scn.set_pose(mesh_node, pose_gl)
        _, depth = renderer.render(scn, flags=pyrender.RenderFlags.RGBA)
        mask = depth > 0
        color = np.array(COLORS_INT[i % len(COLORS_INT)], dtype=np.float32)
        overlay[mask] = (1.0 - alpha) * overlay[mask] + alpha * color

    renderer.delete()

    img = Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8))

    if annotation_text:
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.load_default(size=28)
        except TypeError:
            # Older Pillow: load_default() takes no `size` argument.
            font = ImageFont.load_default()

        margin, pad = 12, 8
        text_box = draw.textbbox((margin, margin), annotation_text, font=font)
        draw.rectangle(
            [text_box[0] - pad, text_box[1] - pad, text_box[2] + pad, text_box[3] + pad],
            fill=(0, 0, 0),
        )
        draw.text((margin, margin), annotation_text, font=font, fill=(255, 255, 255))

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ── Flask endpoint ────────────────────────────────────────────────────────────

@app.post("/estimate")
def estimate():
    # ── Validate inputs ───────────────────────────────────────────────────────
    if "rgb" not in request.files or "depth" not in request.files:
        return "Missing 'rgb' or 'depth' field in request", 400

    # ── Decode RGB (JPEG → uint8 H×W×3) ──────────────────────────────────────
    try:
        arr = np.frombuffer(request.files["rgb"].read(), dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError("cv2.imdecode returned None")
        rgb_array = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        return f"Failed to decode RGB: {e}", 400

    # ── Decode depth (numpy bytes → float32 H×W mm) ───────────────────────────
    try:
        depth_array = np.load(io.BytesIO(request.files["depth"].read())).astype(np.float32)
    except Exception as e:
        return f"Failed to decode depth: {e}", 400

    # Persistent per-request directory (kept on disk for debugging — see REQUESTS_DIR).
    req_id  = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    req_dir = os.path.join(REQUESTS_DIR, req_id)
    os.makedirs(req_dir, exist_ok=True)
    log.info(f"[{req_id}] Request files will be saved to {req_dir}")

    rgb_tmp   = os.path.join(req_dir, "frame.png")
    depth_tmp = os.path.join(req_dir, "frame.npy")
    cnos_tmp  = os.path.join(req_dir, "detections.json")

    # Write files for run_demo to consume (live_demo_inference.py is read-only; it needs paths).
    Image.fromarray(rgb_array).save(rgb_tmp)
    np.save(depth_tmp, depth_array)

    # ── CNOS segmentation ─────────────────────────────────────────────────────
    # Timed stages only (CNOS + B2TF) — excludes demo-harness overhead
    # (decode/file-I/O/rendering) that wouldn't exist in a real deployment.
    t_cnos_start = time.perf_counter()
    try:
        detections = run_cnos_segmentation(
            rgb_array,
            num_max_dets=int(_config.get("cnos_num_max_dets", 10)),
            conf_threshold=float(_config.get("cnos_conf_threshold", 0.30)),
        )
    except Exception as e:
        log.exception("CNOS segmentation failed")
        return f"CNOS error: {e}", 500
    t_cnos = time.perf_counter() - t_cnos_start

    if not detections:
        return "No objects detected above confidence threshold", 400

    with open(cnos_tmp, "w") as f:
        json.dump(detections, f)

    # ── B2TF pose estimation ──────────────────────────────────────────────────
    t_b2tf_start = time.perf_counter()
    try:
        poses = _b2tf.run_demo(
            rgb_path=rgb_tmp,
            depth_path=depth_tmp,
            cnos_json_path=cnos_tmp,
            K=_K,
            depth_scale=float(_config.get("depth_scale", 1.0)),
            output_dir=req_dir,
        )
    except Exception as e:
        log.exception("B2TF run_demo failed")
        return f"Pose estimation error: {e}", 500
    t_b2tf = time.perf_counter() - t_b2tf_start

    pipeline_time_s = t_cnos + t_b2tf
    log.info(f"[{req_id}] Pipeline time — CNOS: {t_cnos:.2f}s, B2TF: {t_b2tf:.2f}s, "
             f"total: {pipeline_time_s:.2f}s")

    if not poses:
        log.warning(f"[{req_id}] run_demo returned no poses (None or empty list)")
        return "No pose estimated", 400

    # ── Render overlay ────────────────────────────────────────────────────────
    annotation = None
    if _config.get("show_pipeline_timing_annotation", True):
        annotation = f"Pose estimation pipeline time: {pipeline_time_s:.2f}s"
    try:
        png_bytes = render_overlay(rgb_array, poses, _K, _pyrender_mesh,
                                   annotation_text=annotation)
    except Exception as e:
        log.exception("Overlay rendering failed")
        return f"Render error: {e}", 500

    with open(os.path.join(req_dir, "overlay.png"), "wb") as f:
        f.write(png_bytes)

    return Response(png_bytes, mimetype="image/png")


# ── Startup ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="B2TF RGBD live demo server")
    parser.add_argument("--config", default="./configs/demo_live.yaml")
    args = parser.parse_args()

    _config = load_config_with_overrides(args.config, argparse.Namespace())
    _K = parse_intrinsics(_config["camera_intrinsics"])

    vizs = _config.get("visualizations", [])
    if "no_visualization" not in vizs:
        log.warning(
            "One or more visualizations are enabled — plt.show() will block the server. "
            "Set 'visualizations: [no_visualization]' in demo_live.yaml."
        )

    log.info("Loading mesh...")
    _mesh_trimesh = trimesh.load(_config["model_path"], force="mesh")
    _pyrender_mesh = pyrender.Mesh.from_trimesh(_mesh_trimesh, smooth=False)
    log.info("Mesh loaded.")

    log.info("Initializing CNOS (FastSAM + DINOv2)...")
    init_cnos(_config)

    log.info("Initializing B2TF (DINOv3 + MegaPose + TeaserPP)...")
    _b2tf = LiveDemoPoseEstimator(_config)
    log.info("B2TF ready. Server starting...")

    # threaded=False: GPU models are not re-entrant, serialize all requests.
    app.run(
        host=_config.get("server_host", "0.0.0.0"),
        port=int(_config.get("server_port", 5000)),
        threaded=False,
        debug=False,
    )
