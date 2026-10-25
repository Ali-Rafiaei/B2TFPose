"""CNOS detection score/count filtering and RLE mask decoding."""

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
