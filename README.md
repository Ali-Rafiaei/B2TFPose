# B2TFPose

Official implementation of **"Back to the Feature: Zero-Shot 6DoF Pose Estimation via
Dense Local Features"** (Ali Rafiaei, Michael Greenspan), accepted to BMVC.
Preprint: https://arxiv.org/abs/2609.06726

B2TFPose estimates the 6DoF pose of a novel object (never seen during training) from a
single RGB image and its 3D CAD model, with zero task-specific training. The only
learned component is a single frozen DINOv3 vision transformer, used both for global
template retrieval and for dense per-patch local feature matching. Classical RANSAC+PnP
turns the resulting correspondences into a pose, the same way SIFT correspondences fed
PnP/Structure-from-Motion in the pre-deep-learning era.

This repository also includes B2TFPose-RGBD, a depth-using variant that builds 3D-3D
correspondences and solves with RANSAC + Horn's method instead of PnP, refined with
ICP/TeaserPP, plus a live demo: a depth camera feeds a server, which returns a pose
overlay.

Three contributions (see the paper for detail):

- **Geodesic-NMS template retrieval.** Diversifies the retrieved templates by enforcing
  a minimum inter-view rotation distance, instead of raw top-k by cosine similarity.
- **Render-guided Re-Correspondence (RRC).** Re-renders the object at the initial pose
  estimate (plus 9 rotation variants), re-encodes with the same frozen backbone, and
  re-solves PnP to sharpen the estimate, with no additional learned parameters.
- **Multi-mask hypothesis selection.** Runs the full pipeline on every plausible
  segmentation mask, not just the top-1, and picks the best resulting pose.

## Repository layout

```
RGB/                   RGB-only pipeline: 2D-3D correspondences, RANSAC-PnP, MegaPose RGB refinement
RGBD/                  RGBD pipeline: 3D-3D correspondences, RANSAC-Horn, ICP/TeaserPP,
                       MegaPose RGBD refinement, and the live demo (live_demo_server.py / live_demo_client.py)
template_generation/   BlenderProc-based template rendering (offline stage, once per object)
cnos/                  git submodule (our fork), zero-shot segmentation
megapose/              git submodule (our fork), render-and-compare pose refinement
predefined_template_poses/   Icosphere-level-2 + in-plane-rotation camera poses used for template rendering
```

## Setup

### 1. Clone with submodules

```
git clone --recurse-submodules <this repo's URL>
# or, if already cloned:
git submodule update --init --recursive
```

### 2. Python environment

```
pip install -r requirements.txt
# See the file for a few packages that need special handling (the CUDA
# build of torch/torchvision, faiss via conda, TEASER++ built from source).
```

### 3. External dependencies not vendored here

- **DINOv3.** Clone [`facebookresearch/dinov3`](https://github.com/facebookresearch/dinov3)
  and download its weights per its own instructions, then point `dinov3_repo_dir` in each
  config at your clone. Subject to Meta's own DINO license; read it before use (see `LICENSE`).
- **BlenderProc** (`pip install blenderproc`). Only needed for template generation
  (step below), invoked as an external tool (`blenderproc run ...`), never imported.
  GPL-3.0; see `LICENSE`.
- **BOP Toolkit** ([`thodan/bop_toolkit`](https://github.com/thodan/bop_toolkit)). For
  AR evaluation of the output CSVs. Not needed to produce pose estimates, only to score them.

## Data setup

### BOP dataset

Download your target dataset (e.g. TUD-L, LM-O, YCB-V) from the
[BOP Benchmark](https://bop.felk.cvut.cz/datasets/) in its standard layout
(`<dataset_root>/models/`, `<dataset_root>/test/<scene>/{rgb,depth,mask_visib}`).

### Templates

Render templates for your object(s) once (offline stage):

```
# edit dataset_root at the top of render() in render_bop_templates.py first.
# cad_name is derived from it automatically (models_cad for T-LESS, else models)
python template_generation/render_bop_templates.py
```

This produces `<dataset_root>/new_templates/level2_icosphere/<obj_id>/...`. Point each
config's `templates_dir` at it.

### CNOS detections (for `bop_inference.py` benchmark runs)

The benchmark pipelines (`RGB/bop_inference.py`, `RGBD/bop_inference.py`) don't run CNOS
live. They read a precomputed detections file. For the 7 BOP core datasets, BOP provides
ready-made CNOS (FastSAM) detections:
[bop-benchmark/bop_extra on Hugging Face](https://huggingface.co/datasets/bop-benchmark/bop_extra/tree/main/default_detections/classic_bop23_model_based_unseen/cnos-fastsam).
Download the file for your dataset, rename it to `cnos_segs_reformatted.json`, and place
it at `<dataset_root>/cnos_segmentations/cnos_segs_reformatted.json`.

The live demo is different: it runs CNOS live, in-process, through the `cnos/`
submodule. See below.

## Usage

### Benchmark evaluation (RGB)

```
python RGB/bop_inference.py --config RGB/configs/default.yaml --obj_id <N> \
    --results_dir <output dir> --log_estimates_csv True
```

### Benchmark evaluation (RGBD)

```
python RGBD/bop_inference.py --config RGBD/configs/default_config_tudl.yaml --obj_id <N> \
    --results_dir <output dir> --log_estimates_csv True
```

(`configs/default_config_lmo.yaml` for LM-O.) Output CSVs are in BOP19 format; score them
with the official `bop_toolkit`.

### Live demo (RGBD)

The live demo pairs a server (runs CNOS + B2TFPose-RGBD) with a client (captures a frame,
sends it, displays the pose overlay). The reference client and `capture_intrinsics.py`
assume a Zivid structured-light camera. The server's `POST /estimate` HTTP API itself is
hardware-agnostic, so write your own client against it for other sensors.

```
# on the machine with the GPU:
python RGBD/live_demo_server.py --config RGBD/configs/demo_live.yaml

# on the machine with the camera:
python RGBD/capture_intrinsics.py        # once, to fill in camera_intrinsics above
python RGBD/live_demo_client.py --server http://<server-ip>:5000
```

## License

MIT (see `LICENSE`) for this repository's own code. `cnos/` and `megapose/` are separate
git submodules under their own licenses (MIT and Apache 2.0 respectively). See `LICENSE`
for the full third-party breakdown, including DINOv3's and BlenderProc's terms as
external (non-vendored) dependencies.
