import csv
import os

import numpy as np

_BOP_COLUMNS = ["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]


def rle_to_mask(rle) -> np.ndarray:
    """Decode a CNOS/BOP run-length-encoded mask dict to a uint8 binary array."""
    h, w = rle["size"]
    mask = np.empty(h * w, dtype=bool)
    idx, parity = 0, False
    for count in rle["counts"]:
        mask[idx : idx + count] = parity
        idx += count
        parity ^= True
    return mask.reshape(h, w, order="F").astype(np.uint8)


def write_pose_to_csv_bop(
    csv_path,
    scene_id,
    im_id,
    obj_id,
    R,
    t,
    score: float = 1.0,
    time: float = -1.0,
):
    """
    Appends one pose to a CSV file in BOP format.
    The CSV file will have columns: scene_id, im_id, obj_id, score, R, t, time
    where R is a flattened 3x3 rotation matrix and t is a 3x1 translation vector.
    Pass score= to record the detection confidence (e.g. CNOS mask score).
    The time column is kept for BOP-format compatibility only; this pipeline
    does not measure wall-clock time, so it is always written as -1.0 (unknown).
    """
    R_str = " ".join(f"{val:.6f}" for val in R.flatten())
    t_str = " ".join(f"{val:.6f}" for val in t.flatten())
    row = [scene_id, im_id, obj_id, score, R_str, t_str, time]

    write_header = not os.path.exists(csv_path)
    if write_header:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(_BOP_COLUMNS)
        writer.writerow(row)
