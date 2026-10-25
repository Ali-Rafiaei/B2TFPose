"""perform_icp is the production point-to-point ICP refinement, used by
mask_pipeline.py. Everything else here (refinement_by_rotation and its
rot_x/y/z helpers) is an unused, exploratory multi-angle ICP strategy —
kept intentionally, not imported anywhere."""

import time

import open3d as o3d
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def perform_icp(object_pc, mesh_points, estimated_pose):
    """Point-to-point ICP: aligns mesh_points to object_pc from estimated_pose.
    Returns (refined_mesh_pointcloud, refined_4x4_transformation)."""
    if isinstance(mesh_points, o3d.geometry.PointCloud):
        mesh_as_pc = mesh_points
    else:
        mesh_as_pc = o3d.geometry.PointCloud()
        mesh_as_pc.points = o3d.utility.Vector3dVector(mesh_points)

    if isinstance(object_pc, np.ndarray):
        object_pc_points = object_pc.copy()
        object_pc = o3d.geometry.PointCloud()
        object_pc.points = o3d.utility.Vector3dVector(object_pc_points)

    transferred_mesh_points = np.dot(mesh_points, estimated_pose[0:3, :3].T) + estimated_pose[0:3, 3].T
    estimated_pc = o3d.geometry.PointCloud()
    estimated_pc.points = o3d.utility.Vector3dVector(transferred_mesh_points)

    # setting the ICP parameters:
    euclidean_distance_bf_icp = np.mean(object_pc.compute_point_cloud_distance(estimated_pc))
    threshold = euclidean_distance_bf_icp

    estimation_method = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    initial_transformation = estimated_pose
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100)

    p2p_registration = o3d.pipelines.registration.registration_icp(mesh_as_pc, object_pc, threshold,
                                                                   initial_transformation,
                                                                   estimation_method, criteria)

    transfered_mesh_with_icp = mesh_as_pc
    transfered_mesh_with_icp.transform(p2p_registration.transformation)

    return transfered_mesh_with_icp, p2p_registration.transformation

def rot_x(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle),  np.cos(angle)]
    ])

def rot_y(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [ np.cos(angle), 0, np.sin(angle)],
        [0, 1, 0],
        [-np.sin(angle), 0, np.cos(angle)]
    ])

def rot_z(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle),  np.cos(angle), 0],
        [0, 0, 1]
    ])
def run_icp_with_rotation(rotation, object_pc, mesh_points, base_transformation):
    rotated_icp_transformation = np.eye(4)
    rotated_icp_transformation[0:3, 0:3] = np.dot(base_transformation[0:3, 0:3], rotation)
    rotated_icp_transformation[0:3, 3] = base_transformation[0:3, 3]

    refined_mesh, refined_transformation = perform_icp(object_pc, mesh_points, rotated_icp_transformation)
    # Use chamfer distance (bidirectional) instead of unidirectional distance.
    # Unidirectional (query→model) is unreliable under partial views since every
    # visible point can find a nearby model point even with a shifted/rotated pose.
    dist_q2m = np.mean(object_pc.compute_point_cloud_distance(refined_mesh))
    dist_m2q = np.mean(refined_mesh.compute_point_cloud_distance(object_pc))
    chamfer_error = (dist_q2m + dist_m2q) / 2.0
    return chamfer_error, refined_transformation

def refinement_by_rotation(object_points, initial_estimated_pose, mesh_points):
    object_pc = o3d.geometry.PointCloud()
    object_pc.points = o3d.utility.Vector3dVector(object_points)

    initial_icp_mesh, transformation = perform_icp(object_pc, mesh_points, initial_estimated_pose)
    dist_q2m = np.mean(object_pc.compute_point_cloud_distance(initial_icp_mesh))
    dist_m2q = np.mean(initial_icp_mesh.compute_point_cloud_distance(object_pc))
    euclidean_distance_wout_refinement = (dist_q2m + dist_m2q) / 2.0

    R_x_180 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    R_y_180 = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    R_z_180 = np.array([[-1, 0, 0], [0, -1, 0], [0, 0, 1]])
    R_x_90_pos = rot_x(90)
    R_x_90_neg = rot_x(-90)
    R_y_90_pos = rot_y(90)
    R_y_90_neg = rot_y(-90)
    R_z_90_pos = rot_z(90)
    R_z_90_neg = rot_z(-90)

    rotations = np.array([np.eye(3, 3), R_x_180, R_y_180, R_z_180, R_x_90_pos, R_y_90_pos, R_z_90_pos, R_x_90_neg, R_y_90_neg, R_z_90_neg])
    # rotations = np.array([np.eye(3, 3), R_x_180, R_y_180, R_z_180])

    errors = np.zeros(len(rotations))
    icp_transformations = np.zeros((len(rotations), 4, 4))

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(run_icp_with_rotation, rot, object_pc, mesh_points, transformation) for rot in rotations]
        results = [future.result() for future in futures]

    errors = np.array([res[0] for res in results])
    icp_transformations = np.array([res[1] for res in results])

    lowest_refinement_error_index = np.argmin(errors)

    if errors[lowest_refinement_error_index] < euclidean_distance_wout_refinement:
        # if lowest_refinement_error_index == 0:
        #     print("Double ICP refinement")
        # else:
        #     if lowest_refinement_error_index < 4:
        #         print(
        #               f"180 degree rotation refinement along {['x', 'y', 'z'][lowest_refinement_error_index-1]} axis applied")
        #     elif lowest_refinement_error_index < 7:
        #         print(
        #             f"+90 degree rotation refinement along {['x', 'y', 'z'][lowest_refinement_error_index-4]} axis applied")
        #     else:
        #         print(
        #             f"-90 degree rotation refinement along {['x', 'y', 'z'][lowest_refinement_error_index-7]} axis applied")

        best_transformation = icp_transformations[lowest_refinement_error_index]
        final_error = errors[lowest_refinement_error_index]
    else:
        # print("No refinement needed")
        best_transformation = transformation
        final_error = euclidean_distance_wout_refinement


    final_transferred_mesh_points = np.dot(mesh_points, best_transformation[0:3, 0:3].T) + best_transformation[0:3, 3].T
    final_transferred_mesh = o3d.geometry.PointCloud()
    final_transferred_mesh.points = o3d.utility.Vector3dVector(final_transferred_mesh_points)
    return final_transferred_mesh, best_transformation

if __name__ == '__main__':
    object_xyz = np.load("../sample for refinement/xyz.npy")
    estimated_pose = np.load("../sample for refinement/estimated_pose.npy")
    mesh_points = np.load("../data/model_points.npy")
    st_time = time.time()
    for i in range(20):
        refinement_by_rotation(object_xyz, estimated_pose, mesh_points)
    total_time = time.time() - st_time
    print("Total time: ", total_time)
    print("Average time: ", total_time / 20)
