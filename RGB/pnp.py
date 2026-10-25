"""
RANSAC-PnP pose estimator for 2D-3D correspondences.

Drop-in replacement for ransac_horn (from horn.py) when query depth is
unavailable.  Instead of 3D-3D rigid-body fitting (SVD / Horn method), this
solves the Perspective-n-Point problem: given object-frame 3D points and their
corresponding 2D image projections, recover the camera-to-object pose via
cv2.solvePnP.

Interface mirrors ransac_horn:
  ransac_pnp(P_obj, P_img, K, ...) -> list of (4,4) pose matrices
"""

import cv2
import numpy as np

from geometry_3d_utils import rotation_error


def compute_ransac_iterations(
    n_correspondences,
    min_inlier_ratio=0.08,
    confidence=0.999,
    sample_size=3,
    max_iters=15000,
):
    """Compute number of RANSAC iterations needed for given confidence level."""
    r = max(min_inlier_ratio, sample_size / max(n_correspondences, sample_size))
    prob_bad = 1.0 - r**sample_size
    if prob_bad <= 0.0:
        return max_iters
    n = int(np.ceil(np.log(1.0 - confidence) / np.log(prob_bad)))
    return min(n, max_iters)


def ransac_pnp(
    points_3d,
    points_2d,
    K,
    num_hypotheses: int = 10,
    reprojection_threshold: float = 8.0,
    max_iterations: int = 10000,
    confidence: float = 0.999,
    hypo_min_dist: float = 15.0,
    weights=None,
    feature_distances=None,
    soft_score_alpha: float = 0.7,
):
    """
    RANSAC-PnP: robust 6DoF pose from 2D-3D correspondences.

    Args:
        points_3d: (N, 3) or list of (3,) arrays — object-frame 3D points (mm).
        points_2d: (N, 2) or list of (2,) arrays — corresponding query image
                   pixel coordinates.
        K:         (3, 3) camera intrinsic matrix.
        num_hypotheses:        Maximum number of diverse pose hypotheses to return.
        reprojection_threshold: Inlier threshold in pixels.
        max_iterations:        RANSAC iteration budget.
        confidence:            Target RANSAC confidence for adaptive stopping.
        hypo_min_dist:         Minimum rotation distance (degrees) between
                               returned hypotheses (enforces diversity).
        weights:               (N,) float32 PROSAC-style sampling weights
                               (higher weight = sampled earlier). None → uniform.
        feature_distances:     (N,) float32 raw L2 distances from FAISS per
                               correspondence. Used to compute a visual quality
                               term in the soft inlier score. None → visual
                               quality is uniform (soft geo scoring only).
        soft_score_alpha:      Geometry weight in [0, 1]. 1.0 = pure geometric
                               soft score; 0.0 = pure visual quality. Default 0.7.

    Returns:
        List of (4, 4) float64 pose matrices (object-to-camera), best-first by
        soft score.  Returns [] if fewer than 4 correspondences supplied.
    """
    pts3 = np.asarray(points_3d, dtype=np.float32)
    pts2 = np.asarray(points_2d, dtype=np.float32)

    if pts3.ndim == 1:
        pts3 = pts3.reshape(-1, 3)
    if pts2.ndim == 1:
        pts2 = pts2.reshape(-1, 2)

    N = len(pts3)
    if N < 4:
        return []

    # ── Weighted sampling probabilities ──────────────────────────────────────
    # Every correspondence can be sampled at every iteration; higher-weight
    # ones are drawn more often. No growing pool — avoids excluding low-weight
    # correspondences when early stopping fires before the pool reaches full size.
    if weights is not None:
        w = np.clip(np.asarray(weights, dtype=np.float64), 0.0, None)
        w_sum = w.sum()
        sample_probs = w / w_sum if w_sum > 0 else None
    else:
        sample_probs = None

    # Keep float64 copies for cv2; normalize 3D to meters for numerical stability
    # (mm-scale 3D + pixel-scale 2D in the same linear system causes poor conditioning)
    pts3_d = pts3.astype(np.float64) / 1000.0   # mm → m
    pts2_d = pts2.astype(np.float64)
    K_d = np.asarray(K, dtype=np.float64)

    # ── Visual quality (z-normalised feature distances) ───────────────────────
    # sigmoid(-z): low distance (similar features) → value near 1.0
    #              high distance (dissimilar)       → value near 0.0
    # When feature_distances is None, vis_quality = 1 for all → soft geo only.
    if feature_distances is not None:
        fd = np.asarray(feature_distances, dtype=np.float64)
        z = (fd - fd.mean()) / (fd.std() + 1e-6)
        vis_quality = 1.0 / (1.0 + np.exp(z))   # sigmoid(-z)
    else:
        vis_quality = np.ones(N, dtype=np.float64)

    # ── Per-iteration records ─────────────────────────────────────────────────
    # columns: [soft_score, n_inliers, mean_reprojection_error]
    votes = np.zeros((max_iterations, 3), dtype=np.float64)
    votes[:, 2] = np.inf
    rvecs = [None] * max_iterations
    tvecs = [None] * max_iterations

    best_inlier_ratio = 0.0
    min_iters_floor = 500
    actual_iters = max_iterations

    for it in range(max_iterations):
        idx = np.random.choice(N, 3, replace=False, p=sample_probs)
        p3 = pts3_d[idx]  # (3, 3)
        p2 = pts2_d[idx]  # (3, 2)

        # P3P: exact closed-form solver for the minimal 3-point case.
        # Returns up to 4 candidate (rvec, tvec) pairs — pick best by inlier count.
        ok, rvecs_out, tvecs_out = cv2.solveP3P(p3, p2, K_d, None, flags=cv2.SOLVEPNP_P3P)
        if not ok or not rvecs_out:
            continue

        rvec, tvec = None, None
        best_n_p3p = -1
        for rv, tv in zip(rvecs_out, tvecs_out):
            proj_p3p, _ = cv2.projectPoints(pts3_d, rv, tv, K_d, None)
            err_p3p = np.linalg.norm(
                proj_p3p.reshape(-1, 2).astype(np.float32) - pts2, axis=1
            )
            n_p3p = int((err_p3p < reprojection_threshold).sum())
            if n_p3p > best_n_p3p:
                best_n_p3p, rvec, tvec = n_p3p, rv, tv

        # Count inliers and compute combined soft score over all N points
        proj, _ = cv2.projectPoints(pts3_d, rvec, tvec, K_d, None)
        proj = proj.reshape(-1, 2).astype(np.float32)
        err = np.linalg.norm(proj - pts2, axis=1)  # (N,)
        inlier_mask = err < reprojection_threshold
        n_inliers = int(inlier_mask.sum())
        mean_err = float(err[inlier_mask].mean()) if n_inliers > 0 else np.inf

        # geo_quality: smooth linear decay from 1 (err=0) to 0 (err=threshold)
        geo_quality = np.clip(1.0 - err / reprojection_threshold, 0.0, 1.0)
        soft_score = float(
            ((geo_quality ** soft_score_alpha) * (vis_quality ** (1.0 - soft_score_alpha))).sum()
        )

        votes[it] = [soft_score, n_inliers, mean_err]
        rvecs[it] = rvec
        tvecs[it] = tvec

        # ── Adaptive early stopping — uses hard inlier count for convergence ──
        inlier_ratio = n_inliers / N
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            if best_inlier_ratio > 0.05 and n_inliers >= 4 and it >= min_iters_floor:
                p_all_out = max(1e-9, 1.0 - best_inlier_ratio ** 3)
                iters_needed = np.log(1.0 - confidence) / np.log(p_all_out)
                if it >= iters_needed * 1.2:
                    actual_iters = it + 1
                    break

    # ── Select diverse top hypotheses ─────────────────────────────────────────
    # Sort: primary soft score descending, secondary inlier count descending,
    #       tertiary mean error ascending
    sort_order = np.lexsort((
        votes[:actual_iters, 2],    # tertiary: mean_err ascending
        -votes[:actual_iters, 1],   # secondary: n_inliers descending
        -votes[:actual_iters, 0],   # primary:   soft_score descending
    ))

    selected_poses = []
    selected_Rs = []

    for rank_idx in sort_order:
        n_inliers = int(votes[rank_idx, 1])
        if n_inliers < 3:
            break  # remaining hypotheses have too few geometric inliers — give up

        rvec = rvecs[rank_idx]
        tvec = tvecs[rank_idx]
        if rvec is None:
            continue

        R_cand, _ = cv2.Rodrigues(rvec)

        # Diversity check: skip if too close to an already-accepted hypothesis
        too_close = any(
            rotation_error(R_cand, R_sel) < hypo_min_dist for R_sel in selected_Rs
        )
        if too_close:
            continue

        # ── Inlier refinement (two-pass, same spirit as horn.py) ─────────────
        proj, _ = cv2.projectPoints(pts3_d, rvec, tvec, K_d, None)
        proj = proj.reshape(-1, 2).astype(np.float32)
        err = np.linalg.norm(proj - pts2, axis=1)
        inlier_mask = err < reprojection_threshold

        if inlier_mask.sum() >= 4:
            ok, rvec_ref, tvec_ref = cv2.solvePnP(
                pts3_d[inlier_mask],
                pts2_d[inlier_mask],
                K_d,
                None,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if ok and rvec_ref is not None:
                rvec, tvec = rvec_ref, tvec_ref

        R_ref, _ = cv2.Rodrigues(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R_ref
        T[:3, 3] = tvec.ravel() * 1000.0   # m → mm to match rest of pipeline

        selected_poses.append(T)
        selected_Rs.append(R_ref)

        if len(selected_poses) >= num_hypotheses:
            break

    best_soft = float(votes[:actual_iters, 0].max()) if actual_iters > 0 else 0.0
    print(
        f"RANSAC-PnP completed {actual_iters} iterations | "
        f"best inlier ratio {best_inlier_ratio:.4f} | "
        f"best soft score {best_soft:.3f} | "
        f"alpha={soft_score_alpha:.2f}"
    )

    return selected_poses
