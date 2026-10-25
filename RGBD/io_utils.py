"""RLE mask decoding and BOP-format CSV output helpers."""

import os
import numpy as np
import pandas as pd


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
    """Appends one pose to a BOP-format CSV (scene_id, im_id, obj_id, score, R, t, time)."""
    R_str = " ".join(f"{val:.6f}" for val in R.flatten())
    t_str = " ".join(f"{val:.6f}" for val in t.flatten())
    row = {
        "scene_id": scene_id,
        "im_id": im_id,
        "obj_id": obj_id,
        "score": score,
        "R": R_str,
        "t": t_str,
        "time": time,
    }
    if not os.path.exists(csv_path):
        df = pd.DataFrame(
            columns=["scene_id", "im_id", "obj_id", "score", "R", "t", "time"]
        )
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(csv_path, index=False)

def save_chosen_mask_idx(mask_idx, img_id, scene_id, obj_id,
                         results_root):
    """Appends which mask index was selected for one image (multi-mask debug log)."""
    output_csv_path = os.path.join(
        results_root, "chosen_masks_in_multimask_setting.csv"
    )
    row = {
        "scene_id": scene_id,
        "im_id": img_id,
        "obj_id": obj_id,
        "mask_idx": mask_idx,
            }
    if not os.path.exists(output_csv_path):
        df = pd.DataFrame(columns=["scene_id", "im_id", "obj_id", "mask_idx"])
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df.to_csv(output_csv_path, index=False)
    else:
        df = pd.read_csv(output_csv_path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
        df.to_csv(output_csv_path, index=False)
