"""CNOS detection loading, score/count filtering, and RLE mask decoding."""

import json
from typing import List, Union

import numpy as np

from io_utils import rle_to_mask


def select_cnos_detections(
    detections_sorted_by_score: List[dict],
    score_threshold: float,
    num_masks: Union[int, str],
) -> List[dict]:
    """Keeps detections at or above score_threshold (falling back to the top
    detection if none clear it), capped to num_masks ("all" or an int).
    Input must already be sorted by score, descending."""
    if not detections_sorted_by_score:
        return []

    acceptable = [
        d for d in detections_sorted_by_score if d["score"] >= score_threshold
    ]
    if not acceptable:
        # Fallback: always return at least the top-scoring detection
        acceptable = detections_sorted_by_score[:1]

    if num_masks == "all":
        return acceptable
    return acceptable[: int(num_masks)]


def decode_masks(detections: List[dict]):
    """Decode a list of CNOS detection dicts into (masks, scores) lists."""
    masks = [rle_to_mask(d["segmentation"]).astype(np.uint8) for d in detections]
    scores = [float(d["score"]) for d in detections]
    return masks, scores


def load_cnos_json(cnos_json_path: str) -> List[dict]:
    """Loads a per-image CNOS detections JSON, keeping only detections with a
    segmentation, sorted by score descending. Also unwraps save_json_bop23-style
    output, which wraps detections as [detections, detections_with_distribution]."""
    with open(cnos_json_path) as f:
        cnos_data = json.load(f)

    if (
        isinstance(cnos_data, list)
        and len(cnos_data) >= 1
        and isinstance(cnos_data[0], list)
    ):
        detections = cnos_data[0]
    else:
        detections = cnos_data

    detections = [d for d in detections if d.get("segmentation")]
    detections = sorted(detections, key=lambda d: d["score"], reverse=True)
    return detections
