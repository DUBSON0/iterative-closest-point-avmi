import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import re  # Added for robust parsing, standard library
import copy  # For deep copying point clouds

def parse_line_lidar_data(data):
    # More robust splitting: handle spaces, semicolons, commas
    elements = re.split(r'[ ;,]+', data.strip())
    if len(elements) < 2:
        raise ValueError("Line does not contain sufficient data fields.")
    
    timestamp = elements[0]
    try:
        float_values = [float(v) for v in elements[1:] if v]  # Skip empty
        if len(float_values) % 3 != 0:
            raise ValueError("Number of values does not form complete xyz triplets.")
        points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])
    except ValueError:
        raise ValueError("Invalid float values in line.")
    
    return timestamp, points

def is_valid_scan(points):
    if len(points) < 10:  # Increased min points for validity
        return False
    if np.all(points == 0):
        return False
    if np.sum(np.abs(points)) < 1e-10:
        return False
    return True

def line_to_pcd(line):
    timestamp, points = parse_line_lidar_data(line)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, timestamp, points

def pairwise_icp(source, target, voxel_size=0.01, coarse_factor=2.0, fine_threshold=0.015, coarse_threshold=0.05, normal_radius=0.1, normal_max_nn=30, robust_k=0.05, min_fitness=0.3, init=np.eye(4)):
    # Coarse stage
    coarse_voxel = voxel_size * coarse_factor
    source_coarse = source.voxel_down_sample(coarse_voxel)
    target_coarse = target.voxel_down_sample(coarse_voxel)
    
    # Estimate normals on target_coarse
    target_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius * coarse_factor, max_nn=normal_max_nn))
    
    # Coarse ICP with point-to-plane and robust kernel
    loss = o3d.pipelines.registration.TukeyLoss(k=robust_k * coarse_factor)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_coarse = o3d.pipelines.registration.registration_icp(
        source_coarse, target_coarse, coarse_threshold,
        init,
        est,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100000,
            relative_fitness=1e-10,
            relative_rmse=1e-10
        )
    )
    
    # Fine stage
    source_fine = source.voxel_down_sample(voxel_size)
    target.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    
    loss_fine = o3d.pipelines.registration.TukeyLoss(k=robust_k)
    est_fine = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_fine)
    reg_fine = o3d.pipelines.registration.registration_icp(
        source_fine, target, fine_threshold,
        reg_coarse.transformation,
        est_fine,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100000,
            relative_fitness=1e-10,
            relative_rmse=1e-10
        )
    )
    
    # Check fitness and fallback if necessary
    if reg_fine.fitness < min_fitness:
        print(f"Warning: Low fitness ({reg_fine.fitness:.3f}) in pairwise ICP, using coarse transformation.")
        transformation = reg_coarse.transformation
        eval_reg = o3d.pipelines.registration.evaluate_registration(source_fine, target, fine_threshold, transformation)
        fitness = eval_reg.fitness
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source_fine, target, fine_threshold, transformation)
    else:
        transformation = reg_fine.transformation
        fitness = reg_fine.fitness
        information = o3d.pipelines.registration.get_information_matrix_from_point_clouds(source_fine, target, fine_threshold, transformation)
    
    return transformation, information, fitness

def process_scan(scan_pcd, global_pose, map_cloud, voxel_size=0.01, max_map_points=3000, coarse_factor=2.0, fine_threshold=0.015, coarse_threshold=0.05, normal_radius=0.1, normal_max_nn=30, robust_k=0.05, min_fitness=0.3):
    # Remove statistical outliers from scan
    scan_pcd, _ = scan_pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    
    if len(scan_pcd.points) < 10:
        return global_pose, map_cloud  # Skip if too few points after filtering
    
    # Handle initial map initialization
    if len(map_cloud.points) == 0:
        scan_downsampled = scan_pcd.voxel_down_sample(voxel_size)
        map_cloud += scan_downsampled
        map_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
        return global_pose, map_cloud
    
    # Multi-stage ICP: Coarse then Fine
    # Coarse stage
    coarse_voxel = voxel_size * coarse_factor
    scan_coarse = scan_pcd.voxel_down_sample(coarse_voxel)
    map_coarse = map_cloud.voxel_down_sample(coarse_voxel)
    
    # Estimate normals on map_coarse (target)
    map_coarse.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius * coarse_factor, max_nn=normal_max_nn))
    
    # Coarse ICP with point-to-plane and robust kernel
    loss = o3d.pipelines.registration.TukeyLoss(k=robust_k * coarse_factor)
    est = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss)
    reg_coarse = o3d.pipelines.registration.registration_icp(
        scan_coarse, map_coarse, coarse_threshold,
        init=global_pose,
        estimation_method=est,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100000,
            relative_fitness=1e-10,
            relative_rmse=1e-10
        )
    )
    
    # Use coarse transformation as init for fine
    fine_init = reg_coarse.transformation
    
    # Fine stage
    scan_downsampled = scan_pcd.voxel_down_sample(voxel_size)
    
    # For fine, use original map_cloud (already downsampled finely)
    # Estimate normals on map_cloud if not present (though initialized above)
    if not map_cloud.has_normals():
        map_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    
    # Fine ICP with point-to-plane and robust kernel
    loss_fine = o3d.pipelines.registration.TukeyLoss(k=robust_k)
    est_fine = o3d.pipelines.registration.TransformationEstimationPointToPlane(loss_fine)
    reg_fine = o3d.pipelines.registration.registration_icp(
        scan_downsampled, map_cloud, fine_threshold,
        init=fine_init,
        estimation_method=est_fine,
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100000,
            relative_fitness=1e-10,
            relative_rmse=1e-10
        )
    )
    
    # Check fitness
    if reg_fine.fitness < min_fitness:
        print(f"Warning: Low fitness ({reg_fine.fitness:.3f}) for scan, using coarse transformation instead.")
        global_pose = reg_coarse.transformation
    else:
        global_pose = reg_fine.transformation
    
    # Transform and add to map
    scan_transformed = scan_downsampled.transform(global_pose)
    map_cloud += scan_transformed
    
    # Downsample map and re-estimate normals
    current_voxel = voxel_size
    map_cloud = map_cloud.voxel_down_sample(current_voxel)
    map_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    
    # Adaptive increase if over max_points
    while len(map_cloud.points) > max_map_points:
        current_voxel *= 1.2
        map_cloud = map_cloud.voxel_down_sample(current_voxel)
        map_cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=normal_max_nn))
    
    return global_pose, map_cloud

def detect_loop(current_pose, keyframe_poses, loop_threshold=5.0):
    for i, pose in enumerate(keyframe_poses):
        dist = np.linalg.norm(current_pose[:3, 3] - pose[:3, 3])
        if dist < loop_threshold and dist > 0.1:  # Avoid self
            return i
    return -1

def build_pose_graph(keyframes, keyframe_poses, voxel_size, coarse_factor, fine_threshold, coarse_threshold, normal_radius, normal_max_nn, robust_k, min_fitness):
    n = len(keyframes)
    pose_graph = o3d.pipelines.registration.PoseGraph()
    
    # Add nodes with initial poses from odometry
    for i in range(n):
        pose_graph.nodes.append(o3d.pipelines.registration.PoseGraphNode(keyframe_poses[i]))
    
    # Add odometry edges (sequential)
    for source_id in range(n - 1):
        target_id = source_id + 1
        init_trans = np.matmul(np.linalg.inv(keyframe_poses[target_id]), keyframe_poses[source_id])
        transformation, information, fitness = pairwise_icp(
            keyframes[source_id], keyframes[target_id], voxel_size, coarse_factor, fine_threshold, coarse_threshold,
            normal_radius, normal_max_nn, robust_k, min_fitness, init=init_trans
        )
        uncertain = fitness < min_fitness
        pose_graph.edges.append(
            o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation, information, uncertain=uncertain)
        )
    
    # Add loop closure edges
    for source_id in range(n):
        for target_id in range(source_id + 2, n):  # Skip adjacent
            dist = np.linalg.norm(keyframe_poses[source_id][:3, 3] - keyframe_poses[target_id][:3, 3])
            if dist < loop_threshold:
                init_trans = np.matmul(np.linalg.inv(keyframe_poses[target_id]), keyframe_poses[source_id])
                transformation, information, fitness = pairwise_icp(
                    keyframes[source_id], keyframes[target_id], voxel_size, coarse_factor, fine_threshold, coarse_threshold,
                    normal_radius, normal_max_nn, robust_k, min_fitness, init=init_trans
                )
                if fitness > min_fitness:
                    pose_graph.edges.append(
                        o3d.pipelines.registration.PoseGraphEdge(source_id, target_id, transformation, information, uncertain=True)
                    )
    
    return pose_graph

def optimize_pose_graph(pose_graph, voxel_size):
    option = o3d.pipelines.registration.GlobalOptimizationOption(
        max_correspondence_distance=voxel_size * 1.5,
        edge_prune_threshold=0.25,
        reference_node=0
    )
    o3d.pipelines.registration.global_optimization(
        pose_graph,
        o3d.pipelines.registration.GlobalOptimizationLevenbergMarquardt(),
        o3d.pipelines.registration.GlobalOptimizationConvergenceCriteria(),
        option
    )
    return pose_graph

def reconstruct_map(keyframes, optimized_poses, voxel_size):
    map_cloud = o3d.geometry.PointCloud()
    for i, pcd in enumerate(keyframes):
        transformed = copy.deepcopy(pcd).transform(optimized_poses[i])
        map_cloud += transformed
    return map_cloud.voxel_down_sample(voxel_size)  # Final downsample

def visualize_trajectory_and_map(pose_trajectory, map_cloud, num_scans):
    pose_trajectory = np.array(pose_trajectory)
    map_points = np.asarray(map_cloud.points)
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2], c='gray', s=1, alpha=0.3, label='Map')
    ax.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], pose_trajectory[:, 2], 'r-', linewidth=3, label='Trajectory')
    ax.scatter(pose_trajectory[0, 0], pose_trajectory[0, 1], pose_trajectory[0, 2], c='g', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax.scatter(pose_trajectory[-1, 0], pose_trajectory[-1, 1], pose_trajectory[-1, 2], c='b', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Pose Trajectory Over Map ({len(map_points)} points, {num_scans} scans)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('trajectory_and_map_3d.png', dpi=150)
    
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    ax2.scatter(map_points[:, 0], map_points[:, 1], c='gray', s=1, alpha=0.3, label='Map')
    ax2.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], 'r-', linewidth=3, label='Trajectory')
    ax2.scatter(pose_trajectory[0, 0], pose_trajectory[0, 1], c='g', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax2.scatter(pose_trajectory[-1, 0], pose_trajectory[-1, 1], c='b', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title(f'Pose Trajectory Over Map - XY Plane ({len(map_points)} points, {num_scans} scans)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    plt.tight_layout()
    plt.savefig('trajectory_and_map_2d.png', dpi=150)
    plt.show()

def run_icp_pipeline(data_file, max_scans=1000, voxel_size=0.01, max_map_points=3000, pose_sample_rate=10, progress_interval=50, coarse_factor=2.0, fine_threshold=0.02, coarse_threshold=0.1, normal_radius=0.1, normal_max_nn=30, robust_k=0.05, min_fitness=0.1, keyframe_interval=10, loop_threshold=5.0):
    global_pose = np.eye(4)
    map_cloud = o3d.geometry.PointCloud()
    pose_trajectory = []
    valid_scan_count = 0
    skipped_scan_count = 0
    keyframes = []  # List of downsampled PCDs in local frame
    keyframe_poses = []  # List of absolute poses
    
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_scans:
                break
            
            try:
                scan_pcd, timestamp, points = line_to_pcd(line)
            except ValueError as e:
                print(f"Skipping line {i+1}: {e}")
                skipped_scan_count += 1
                continue
            
            if not is_valid_scan(points):
                skipped_scan_count += 1
                continue
            
            global_pose, map_cloud = process_scan(
                scan_pcd, global_pose, map_cloud, voxel_size, max_map_points,
                coarse_factor, fine_threshold, coarse_threshold, normal_radius, normal_max_nn, robust_k, min_fitness
            )
            valid_scan_count += 1
            
            # Add to trajectory
            if valid_scan_count % pose_sample_rate == 0:
                pose_trajectory.append(global_pose[:3, 3].copy())
            
            # Add keyframe every interval
            if valid_scan_count % keyframe_interval == 0:
                keyframe = copy.deepcopy(scan_pcd).voxel_down_sample(voxel_size)  # Local frame
                keyframes.append(keyframe)
                keyframe_poses.append(global_pose.copy())
                
                # Detect loop
                loop_id = detect_loop(global_pose, keyframe_poses[:-1], loop_threshold)
                if loop_id != -1:
                    print(f"Loop detected with keyframe {loop_id}")
                    # Re-optimize pose graph periodically
                    pose_graph = build_pose_graph(
                        keyframes, keyframe_poses, voxel_size, coarse_factor, fine_threshold, coarse_threshold,
                        normal_radius, normal_max_nn, robust_k, min_fitness
                    )
                    pose_graph = optimize_pose_graph(pose_graph, voxel_size)
                    # Update poses
                    for j in range(len(keyframes)):
                        keyframe_poses[j] = pose_graph.nodes[j].pose
                    # Reconstruct map from optimized keyframes
                    map_cloud = reconstruct_map(keyframes, keyframe_poses, voxel_size)
            
            if valid_scan_count % progress_interval == 0:
                print(f"Scan {valid_scan_count} (line {i+1}): timestamp {timestamp}")
                print(f"Map size: {len(map_cloud.points)} points")
                print(f"Pose translation: [{global_pose[0,3]:.3f}, {global_pose[1,3]:.3f}, {global_pose[2,3]:.3f}]")
                rotation = np.linalg.norm(global_pose[:3, :3] - np.eye(3))
                print(f"Rotation magnitude: {rotation:.3f}\n")
    
    # Final optimization
    print("Performing final pose graph optimization...")
    pose_graph = build_pose_graph(
        keyframes, keyframe_poses, voxel_size, coarse_factor, fine_threshold, coarse_threshold,
        normal_radius, normal_max_nn, robust_k, min_fitness
    )
    pose_graph = optimize_pose_graph(pose_graph, voxel_size)
    
    # Update trajectory and map
    pose_trajectory = [node.pose[:3, 3] for node in pose_graph.nodes]
    map_cloud = reconstruct_map(keyframes, [node.pose for node in pose_graph.nodes], voxel_size)
    
    print(f"\nFinal results:")
    print(f"Valid scans processed: {valid_scan_count}")
    print(f"Scans skipped: {skipped_scan_count}")
    print(f"Total lines read: {i + 1}")
    print(f"Final map size: {len(map_cloud.points)} points")
    print(f"Final pose:\n{global_pose}")
    
    o3d.io.write_point_cloud("output_map.pcd", map_cloud)
    
    visualize_trajectory_and_map(pose_trajectory, map_cloud, valid_scan_count)
    
    return global_pose, map_cloud, pose_trajectory

if __name__ == "__main__":
    data_file = "data/data_recording_20251024_175931_lidar_reader__Decoded_VLP_to_PointCloud_1_1_XYZ_points.csv"
    max_scans = 3000
    
    global_pose, map_cloud, pose_trajectory = run_icp_pipeline(
        data_file=data_file,
        max_scans=max_scans,
        voxel_size=0.15,
        max_map_points=3000,
        pose_sample_rate=10,
        progress_interval=50,
        coarse_factor=2.0,
        fine_threshold=0.02,
        coarse_threshold=0.1,
        normal_radius=0.1,
        normal_max_nn=30,
        robust_k=0.05,
        min_fitness=0.1,
        keyframe_interval=10,
        loop_threshold=5.0
    )