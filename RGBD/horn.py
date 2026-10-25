from typing import Union
import numpy as np
from numba import jit

from geometry_3d_utils import rotation_error


class HornPoseFitting:
    def __init__(self):
        super(HornPoseFitting, self).__init__()

    def solve_absolute_orientation(self, P1, P2, n, A):
        C1 = np.zeros((3))
        C2 = np.zeros((3))
        Sxx = 0.0
        Sxy = 0.0
        Sxz = 0.0
        Syx = 0.0
        Syy = 0.0
        Syz = 0.0
        Szx = 0.0
        Szy = 0.0
        Szz = 0.0
        N = np.zeros((5, 5))
        R = np.zeros((3, 3))
        T = np.zeros((3))
        # calculate centroids
        for i in range(0, n):
            for j in range(0, 3):
                C1[j] += P1[i][j]
                C2[j] += P2[i][j]

        for j in range(0, 3):
            C1[j] /= n
            C2[j] /= n

        # translate point sets to cetroids
        for i in range(0, n):
            for j in range(0, 3):
                P1[i][j] -= C1[j]
                P2[i][j] -= C2[j]

        # calculate elements of M matrix
        for i in range(0, n):
            Sxx += P1[i][0] * P2[i][0]
            Sxy += P1[i][0] * P2[i][1]
            Sxz += P1[i][0] * P2[i][2]

            Syx += P1[i][1] * P2[i][0]
            Syy += P1[i][1] * P2[i][1]
            Syz += P1[i][1] * P2[i][2]

            Szx += P1[i][2] * P2[i][0]
            Szy += P1[i][2] * P2[i][1]
            Szz += P1[i][2] * P2[i][2]

        # generate N matrix

        # Note : matrix indeces start at 1, to be NR compatible

        N[1][1], N[1][2], N[1][3], N[1][4] = (
            Sxx + Syy + Szz,
            Syz - Szy,
            Szx - Sxz,
            Sxy - Syx,
        )
        N[2][1], N[2][2], N[2][3], N[2][4] = (
            Syz - Szy,
            Sxx - Syy - Szz,
            Sxy + Syx,
            Szx + Sxz,
        )
        N[3][1], N[3][2], N[3][3], N[3][4] = (
            Szx - Sxz,
            Sxy + Syx,
            -Sxx + Syy - Szz,
            Syz + Szy,
        )
        N[4][1], N[4][2], N[4][3], N[4][4] = (
            Sxy - Syx,
            Szx + Sxz,
            Syz + Szy,
            -Sxx - Syy + Szz,
        )

        N_matrix = N[1:5, 1:5]
        # Ensure perfect symmetry
        N_matrix = 0.5 * (N_matrix + N_matrix.T)

        # Use eigh for symmetric matrices (faster & more stable)
        eigvals, eigvecs = np.linalg.eigh(N_matrix)

        max_eigind = np.argmax(eigvals)
        max_eigval = eigvals[max_eigind]

        # Extract real parts to handle numerical noise
        q0 = np.real(eigvecs[0, max_eigind])
        q1 = np.real(eigvecs[1, max_eigind])
        q2 = np.real(eigvecs[2, max_eigind])
        q3 = np.real(eigvecs[3, max_eigind])

        R[0][0] = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
        R[0][1] = 2 * (q1 * q2 - q0 * q3)
        R[0][2] = 2 * (q1 * q3 + q0 * q2)
        R[1][0] = 2 * (q1 * q2 + q0 * q3)
        R[1][1] = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
        R[1][2] = 2 * (q2 * q3 - q0 * q1)
        R[2][0] = 2 * (q1 * q3 - q0 * q2)
        R[2][1] = 2 * (q2 * q3 + q0 * q1)
        R[2][2] = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2

        # calculate translation
        T[0] = C2[0] - (R[0][0] * C1[0] + R[0][1] * C1[1] + R[0][2] * C1[2])
        T[1] = C2[1] - (R[1][0] * C1[0] + R[1][1] * C1[1] + R[1][2] * C1[2])
        T[2] = C2[2] - (R[2][0] * C1[0] + R[2][1] * C1[1] + R[2][2] * C1[2])

        # compose into matrix
        for i in range(0, 3):
            for j in range(0, 3):
                A[i][j] = R[i][j]

        for j in range(0, 3):
            A[j][3] = T[j]
            A[3][j] = 0.0

        A[3][3] = 1.0

        # restore original point sets
        for i in range(0, n):
            for j in range(0, 3):
                P1[i][j] += C1[j]
                P2[i][j] += C2[j]


def compute_ransac_iterations(n_correspondences, min_inlier_ratio=0.08,
                               confidence=0.999, sample_size=3, max_iters=15000):
    """Compute number of RANSAC iterations needed for given confidence level."""
    r = max(min_inlier_ratio, sample_size / max(n_correspondences, sample_size))
    prob_bad = 1.0 - r ** sample_size
    if prob_bad <= 0.0:
        return max_iters
    n = int(np.ceil(np.log(1.0 - confidence) / np.log(prob_bad)))
    return min(n, max_iters)


def _is_degenerate(pts, min_area=1e-6):
    """Check if 3 points are (near-)collinear via cross-product area."""
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[0]
    return np.linalg.norm(np.cross(v1, v2)) < min_area


def ransac_horn(
    P1, P2, threshold=15, max_iterations=300,
    num_hypotheses=1, hypo_min_dist=0, early_stop_mode=None,
    weights=None,
):
    """RANSAC over the Horn absolute-orientation solver: sample 3 points, fit,
    count inliers within threshold, repeat, then return the top num_hypotheses
    transformations refined on their inlier sets."""
    n = len(P1)
    if n < 3:
        # Not enough points to estimate pose.
        return [np.eye(4)]

    # PROSAC-style: sort by weight descending so sampling favours best correspondences
    if weights is not None:
        sorted_order = np.argsort(weights)[::-1]
        P1 = np.asarray(P1)[sorted_order]
        P2 = np.asarray(P2)[sorted_order]
        weights = weights[sorted_order]

    if early_stop_mode == "ratio":
        votes, transformation_list = ransac_horn_ratio_early_stop(
            P1, P2, threshold, max_iterations, confidence=0.99, numba=True,
            weights=weights,
        )
    elif early_stop_mode == "it":
        votes, transformation_list = ransac_horn_iteration_early_stop(
            P1, P2, threshold, max_iterations, numba=True,
            weights=weights,
        )
    elif early_stop_mode is None:
        votes, transformation_list = ransac_horn_no_early_stop(
            P1, P2, threshold, max_iterations, numba=True,
            weights=weights,
        )
    else:
        raise ValueError(f"Invalid early_stop_mode: {early_stop_mode}")
    # Select the best vote based on the highest number of inliers and lowest error
    best_vote_indices = select_best_vote(
        votes,
        transformation_list,
        num_hypotheses=num_hypotheses,
        hypo_min_dist=hypo_min_dist,
    )
    # Select the best vote based on the highest number of inliers only
    # best_vote_indices = select_best_vote_max_inliers(votes, num_hypotheses=num_hypotheses)

    best_transformations = transformation_list[best_vote_indices]

    # Refine the transformation using inliers.
    refinement_threshold = threshold
    refined_transformations = []
    for best_transformation in best_transformations:
        refined_transformation = refit_with_inliers(
            P1, P2, best_transformation, refinement_threshold
        )
        refined_transformations.append(refined_transformation)

    return refined_transformations


def refit_with_inliers(P1, P2, transformation, threshold):
    """
    Refine transformation using all inliers. Runs two passes so the second fit
    uses the tighter inlier set produced by the first fit.
    """
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)

    for _ in range(2):
        P1_transformed = np.dot(P1, transformation[:3, :3].T) + transformation[:3, 3]
        inliers_mask = np.linalg.norm(P1_transformed - P2, axis=1) < threshold
        if inliers_mask.sum() < 3:
            return transformation
        refined_transformation = np.eye(4)
        horn_fitter = HornPoseFitting()
        horn_fitter.solve_absolute_orientation(
            P1[inliers_mask].copy(), P2[inliers_mask].copy(),
            int(inliers_mask.sum()), refined_transformation,
        )
        transformation = refined_transformation

    return transformation


def select_best_vote(votes, pose_list, num_hypotheses=1, hypo_min_dist=0):
    # Require at least 3 inliers; fall back to best inlier count if none qualify
    valid_mask = votes[:, 0] >= 3
    if not valid_mask.any():
        return [int(np.argmax(votes[:, 0]))]

    avg_errors = votes[:, 1] / (votes[:, 0] + 1e-10)
    # lexsort: primary = inlier count (descending), secondary = avg error (ascending)
    sorted_indices = np.lexsort((avg_errors, -votes[:, 0]))
    sorted_indices = [int(i) for i in sorted_indices if valid_mask[i]]

    if hypo_min_dist > 0 and num_hypotheses > 1:
        filtered_indices = []
        for idx in sorted_indices:
            current_pose = pose_list[idx]
            too_close = False
            for selected_idx in filtered_indices:
                selected_pose = pose_list[selected_idx]
                rot_dist = rotation_error(current_pose, selected_pose)
                if rot_dist < hypo_min_dist:
                    too_close = True
                    break
            if not too_close:
                filtered_indices.append(idx)
            if len(filtered_indices) >= num_hypotheses:
                break
        best_vote_indices = filtered_indices

    else:
        best_vote_indices = sorted_indices[:num_hypotheses]
    # best_vote = votes[best_vote_idx]
    # best_inlier_indices = all_inlier_indices[best_vote_idx]
    # best_inlier_indices = list(best_inlier_indices[np.nonzero(best_inlier_indices)] - 1)

    return best_vote_indices


def select_best_vote_max_inliers(votes, num_hypotheses=1):
    top_vote_indices = np.argsort(-votes[:, 0])[:num_hypotheses]
    return top_vote_indices


def ransac_horn_no_early_stop(P1: np.ndarray, P2: np.ndarray,
                             threshold: Union[int, float],
                             max_iterations: int, numba: bool = True,
                             weights=None):
    """
    RANSAC inner loop with no early stopping and PROSAC-style growing candidate pool.
    Assumes P1/P2 are already sorted by weight descending when weights is not None.
    """
    n = len(P1)
    votes = np.zeros((max_iterations, 2))
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    transformation_list = np.zeros((max_iterations, 4, 4))

    for i in range(max_iterations):
        # PROSAC-style growing pool: starts at 3, expands to n by end
        if weights is not None:
            pool_size = min(n, max(3, int(3 + (n - 3) * (i / max_iterations) ** 0.5)))
            indices = np.random.choice(pool_size, 3, replace=False)
        else:
            indices = np.random.choice(n, 3, replace=False)

        P1_sample = P1[indices]
        P2_sample = P2[indices]

        if _is_degenerate(P1_sample) or _is_degenerate(P2_sample):
            votes[i] = [0, np.inf]
            transformation_list[i] = np.eye(4)
            continue

        A = np.eye(4)
        horn_fitter = HornPoseFitting()
        horn_fitter.solve_absolute_orientation(P1_sample, P2_sample, 3, A)

        if numba:
            transformed_P1 = transform_points_jit(P1, A[:3, :3], A[:3, 3])
            distances = compute_distances_jit(transformed_P1, P2)
            inliers_count, total_error = count_inliers_jit(distances, threshold)
        else:
            transformed_P1 = np.dot(P1, A[:3, :3].T) + A[:3, 3]
            inliers_count, total_error = compute_inliers_sampled(
                transformed_P1, P2, threshold
            )

        if inliers_count < 3:
            votes[i] = [inliers_count, np.inf]
            transformation_list[i] = A
            continue

        votes[i] = [inliers_count, total_error]
        transformation_list[i] = A

    return votes, transformation_list


def ransac_horn_iteration_early_stop(P1: np.ndarray, P2: np.ndarray,
                                   threshold: Union[int, float],
                                   max_iterations: int, numba: bool = True,
                                   weights=None):
    """Stops early if inlier count hasn't improved for a while. Returns (votes, transformation_list)."""
    n = len(P1)
    votes = np.zeros((max_iterations, 2))
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    transformation_list = np.zeros((max_iterations, 4, 4))

    # Track best result for adaptive stopping.
    best_inlier_count = 0
    iterations_since_improvement = 0

    for i in range(max_iterations):
        indices = np.random.choice(n, 3, replace=False, p=weights)
        P1_sample = P1[indices]
        P2_sample = P2[indices]

        if _is_degenerate(P1_sample) or _is_degenerate(P2_sample):
            votes[i] = [0, np.inf]
            transformation_list[i] = np.eye(4)
            continue

        A = np.eye(4)
        horn_fitter = HornPoseFitting()
        horn_fitter.solve_absolute_orientation(P1_sample, P2_sample, 3, A)

        if numba:
            transformed_P1 = transform_points_jit(P1, A[:3, :3], A[:3, 3])
            distances = compute_distances_jit(transformed_P1, P2)
            inliers_count, total_error = count_inliers_jit(distances, threshold)
        else:
            transformed_P1 = np.dot(P1, A[:3, :3].T) + A[:3, 3]
            inliers_count, total_error = compute_inliers_sampled(
                transformed_P1, P2, threshold
            )

        # Early rejection if fewer than 3 inliers
        if inliers_count < 3:
            votes[i] = [inliers_count, np.inf]
            transformation_list[i] = A
            continue

        votes[i] = [inliers_count, total_error]
        transformation_list[i] = A

        # Track improvements for adaptive stopping
        if inliers_count > best_inlier_count:
            best_inlier_count = inliers_count
            iterations_since_improvement = 0
        else:
            iterations_since_improvement += 1

        # Early stopping if no improvement for 1500 iterations
        if iterations_since_improvement > 1500 and i > 2500:
            print(f"Early stopping RANSAC at iteration {i} / {max_iterations}")
            votes = votes[: i + 1]
            transformation_list = transformation_list[: i + 1]
            break

    return votes, transformation_list


def ransac_horn_ratio_early_stop(
    P1, P2, threshold, max_iterations, confidence=0.99, numba=True,
    weights=None,
):
    """Stops once the best inlier ratio seen makes further improvement
    statistically unlikely at the target confidence. Default early-stop mode
    for ransac_horn. Returns (votes, transformation_list)."""
    n = len(P1)
    votes = []
    P1 = np.asarray(P1)
    P2 = np.asarray(P2)
    transformation_list = []

    best_inlier_ratio = 0
    best_inlier_count = 0

    for i in range(max_iterations):
        indices = np.random.choice(n, 3, replace=False, p=weights)
        P1_sample = P1[indices]
        P2_sample = P2[indices]

        if _is_degenerate(P1_sample) or _is_degenerate(P2_sample):
            votes.append([0, float("inf")])
            transformation_list.append(np.eye(4))
            continue

        A = np.eye(4)
        horn_fitter = HornPoseFitting()
        horn_fitter.solve_absolute_orientation(P1_sample, P2_sample, 3, A)

        if numba:
            transformed_P1 = transform_points_jit(P1, A[:3, :3], A[:3, 3])
            distances = compute_distances_jit(transformed_P1, P2)
            inliers_count, total_error = count_inliers_jit(distances, threshold)

        else:
            transformed_P1 = np.dot(P1, A[:3, :3].T) + A[:3, 3]
            inliers_count, total_error = compute_inliers_sampled(
                transformed_P1, P2, threshold
            )

        if inliers_count < 3:
            votes.append([0, float("inf")])
            transformation_list.append(np.eye(4))
            continue

        votes.append([inliers_count, total_error])
        transformation_list.append(A)

        # Update best inlier ratio
        inlier_ratio = inliers_count / n
        if inlier_ratio > best_inlier_ratio:
            best_inlier_ratio = inlier_ratio
            best_inlier_count = inliers_count

        # Adaptive termination on the best inlier ratio seen so far, guarded by
        # a minimum absolute inlier count and iteration floor (avoids stopping
        # early on a lucky first sample or a spuriously loose threshold).
        min_abs_inliers = max(10, int(0.15 * n))
        min_iter_floor = max(500, int(0.05 * max_iterations))
        if best_inlier_ratio > 0.05 and best_inlier_count >= min_abs_inliers and i >= min_iter_floor:
            epsilon = 1e-9
            prob_all_outliers = max(epsilon, 1 - best_inlier_ratio**3)
            iterations_needed = np.log(1 - confidence) / np.log(prob_all_outliers)

            # Use statistical criterion with 20% safety margin (no arbitrary cap)
            if i >= iterations_needed * 1.2:
                print(f"Adaptive stopping RANSAC at iteration {i} / {max_iterations}")
                print(f"  Best inlier ratio: {best_inlier_ratio:.3f}, needed: {iterations_needed:.0f}")
                break

    votes = np.array(votes)
    transformation_list = np.array(transformation_list)
    return votes, transformation_list


def compute_inliers_sampled(transformed_P1, P2, threshold, full_check_threshold=500):
    """
    Compute inluer count efficiently:
    - If N <= threshold: compute all distances
    - If N > threshold: sample distances, estimate total inliers
    """
    n = len(transformed_P1)

    if n <= full_check_threshold:
        distances = np.linalg.norm(transformed_P1 - P2, axis=1)
        inlier_mask = distances < threshold
        inliers_count = np.sum(inlier_mask)
        total_error = np.sum(distances[inlier_mask])

    else:
        sample_size = full_check_threshold
        sample_indices = np.random.choice(n, sample_size, replace=False)

        sampled_distances = np.linalg.norm(
            transformed_P1[sample_indices] - P2[sample_indices], axis=1
        )
        sampled_inlier_mask = sampled_distances < threshold
        sampled_inliers_count = np.sum(sampled_inlier_mask)
        sampled_total_error = np.sum(sampled_distances[sampled_inlier_mask])

        # Estimate total inliers and error
        inliers_count = int((sampled_inliers_count / sample_size) * n)
        total_error = (sampled_total_error / sample_size) * n

    return inliers_count, total_error


@jit(nopython=True)
def transform_points_jit(points, R, t):
    """Fast point transformation using Numba JIT"""
    n = points.shape[0]
    result = np.empty((n, 3), dtype=np.float64)
    for i in range(n):
        for j in range(3):
            result[i, j] = t[j]
            for k in range(3):
                result[i, j] += R[j, k] * points[i, k]
    return result


@jit(nopython=True)
def compute_distances_jit(points1, points2):
    """fast distance computation using numba jit"""
    n = points1.shape[0]
    distances = np.empty(n, dtype=np.float64)
    for i in range(n):
        dist_sq = 0.0
        for j in range(3):
            diff = points1[i, j] - points2[i, j]
            dist_sq += diff * diff
        distances[i] = np.sqrt(dist_sq)
    return distances


@jit(nopython=True)
def count_inliers_jit(distances, threshold):
    """Fast inlier counting using Numba JIT"""
    count = 0
    total_error = 0.0
    for i in range(len(distances)):
        if distances[i] < threshold:
            count += 1
            total_error += distances[i]
    return count, total_error

if __name__ == "__main__":

    # Generate test point clouds
    np.random.seed(42)
    P1 = np.random.randn(100, 3)

    R_true = np.array(
        [[0.985704243476814, 0.0704034779880539, 0.1530727400163023], 
        [0.00719019718560673, 0.8901083634738218, -0.45569197250878424], 
        [-0.16833421350581995, 0.4502781106287754, 0.8768762072766877]]
    )

    t_true = np.array([1, 2, 3])
    gt_pose = np.eye(4)
    gt_pose[:3, :3] = R_true
    gt_pose[:3, 3] = t_true

    R_true_w_noise = R_true + 0.01 * np.random.randn(3, 3)
    U, _, Vt = np.linalg.svd(R_true_w_noise)
    # Project back to a valid rotation matrix.
    R_true = U @ Vt

    P2 = (P1 @ R_true.T) + t_true

    # Test the Horn solver directly
    A = np.eye(4)
    horn_fitter = HornPoseFitting()
    horn_fitter.solve_absolute_orientation(P1.copy(), P2.copy(), len(P1), A)

    print("Estimated rotation:")
    print(A[:3, :3])
    print("\nTrue rotation:")
    print(R_true)
    print("\nRotation error (degrees):", rotation_error(A, gt_pose))
    print("\nEstimated translation:")
    print(A[:3, 3])
    print("\nTrue translation:")
    print(t_true)
