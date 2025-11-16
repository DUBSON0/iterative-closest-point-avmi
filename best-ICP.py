import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def parse_line_lidar_data(data):
    elements = data.strip().replace(";", " ").split()
    if len(elements) < 2:
        raise ValueError("Line does not contain sufficient data fields.")
    
    timestamp = elements[0]
    float_values = [float(v) for v in elements[1:]]
    if len(float_values) % 3 != 0:
        raise ValueError("Number of values does not form complete xyz triplets.")
    points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])
    return timestamp, points

def is_valid_scan(points, min_points=10):
    if len(points) < min_points:
        return False
    if np.all(points == 0):
        return False
    if np.sum(np.abs(points)) < 1e-10:
        return False
    variance = np.var(points)
    if variance < 1e-10:
        return False
    return True

def line_to_pcd(line):
    timestamp, points = parse_line_lidar_data(line)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, timestamp, points

def process_scan(scan_pcd, global_pose, map_cloud, voxel_size=0.05, max_map_points=5000):
    scan_downsampled = scan_pcd.voxel_down_sample(voxel_size)

    if len(scan_downsampled.points) < 10:
        return global_pose, map_cloud, False

    if len(map_cloud.points) == 0:
        map_cloud += scan_downsampled
        return global_pose, map_cloud, True

    max_correspondence_distance = voxel_size * 3.0

    reg = o3d.pipelines.registration.registration_icp(
        scan_downsampled, map_cloud, max_correspondence_distance,
        init=global_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=50,
            relative_fitness=2,
            relative_rmse=2
        )
    )

    fitness_threshold = 0.1
    rmse_threshold = voxel_size * 3.0

    if reg.fitness < fitness_threshold:
        print(f"  Warning: Low fitness {reg.fitness:.3f} (threshold {fitness_threshold})")
        return global_pose, map_cloud, False

    if reg.inlier_rmse > rmse_threshold:
        print(f"  Warning: High RMSE {reg.inlier_rmse:.3f} (threshold {rmse_threshold:.3f})")
        return global_pose, map_cloud, False

    transformation_magnitude = np.linalg.norm(reg.transformation[:3, 3])
    if transformation_magnitude > 1.0:
        print(f"  Warning: Large transformation jump {transformation_magnitude:.3f}m - rejecting")
        return global_pose, map_cloud, False

    global_pose = reg.transformation

    scan_transformed = scan_downsampled.transform(reg.transformation)
    map_cloud += scan_transformed

    map_cloud = map_cloud.voxel_down_sample(voxel_size)

    if len(map_cloud.points) > max_map_points:
        map_cloud = map_cloud.voxel_down_sample(voxel_size * 1.5)

    return global_pose, map_cloud, True

def visualize_trajectory_and_map(pose_trajectory, map_cloud, num_scans, failed_scans):
    pose_trajectory = np.array(pose_trajectory)
    map_points = np.asarray(map_cloud.points)

    fig = plt.figure(figsize=(15, 6))

    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(map_points[:, 0], map_points[:, 1], map_points[:, 2],
                c='gray', s=1, alpha=0.3, label='Map')
    ax1.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], pose_trajectory[:, 2],
             'r-', linewidth=2, label='Trajectory')
    ax1.scatter(pose_trajectory[0, 0], pose_trajectory[0, 1], pose_trajectory[0, 2],
                c='g', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax1.scatter(pose_trajectory[-1, 0], pose_trajectory[-1, 1], pose_trajectory[-1, 2],
                c='b', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    ax1.set_title(f'3D Map & Trajectory\n{len(map_points)} points, {num_scans} scans')
    ax1.legend()

    ax2 = fig.add_subplot(122)
    ax2.scatter(map_points[:, 0], map_points[:, 1], c='gray', s=1, alpha=0.3, label='Map')
    ax2.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], 'r-', linewidth=2, label='Trajectory')
    ax2.scatter(pose_trajectory[0, 0], pose_trajectory[0, 1],
                c='g', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
    ax2.scatter(pose_trajectory[-1, 0], pose_trajectory[-1, 1],
                c='b', s=150, marker='s', label='End', edgecolors='black', linewidth=2)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')

    success_rate = (num_scans - failed_scans) / num_scans * 100 if num_scans > 0 else 0
    ax2.set_title(f'XY Plane View\nSuccess rate: {success_rate:.1f}%')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')

    plt.tight_layout()
    plt.savefig('trajectory_and_map.png', dpi=150, bbox_inches='tight')
    plt.show()

def run_icp_pipeline(data_file, max_scans=1000, voxel_size=0.05, max_map_points=5000,
                     pose_sample_rate=10, progress_interval=50):
    global_pose = np.eye(4)
    map_cloud = o3d.geometry.PointCloud()
    pose_trajectory = []
    valid_scan_count = 0
    skipped_scan_count = 0
    failed_registration_count = 0

    print("🚀 Starting ICP SLAM Pipeline")
    print(f"   Voxel size: {voxel_size}m")
    print(f"   Max correspondence distance: {voxel_size * 3.0}m")
    print(f"   Max map points: {max_map_points}")
    print("-" * 60)

    return pcd, timestamp

def process_scan(scan_pcd, global_pose, map_cloud, voxel_size=0.01, max_map_points=3000):
    scan_downsampled = scan_pcd.voxel_down_sample(voxel_size)
    
    if len(map_cloud.points) == 0:
        map_cloud += scan_downsampled
        return global_pose, map_cloud
    
    reg = o3d.pipelines.registration.registration_icp(
        scan_downsampled, map_cloud, 0.005,
        init=global_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=100000,
            relative_fitness=1e-15,
            relative_rmse=1e-15
        )
    )
    
    global_pose = reg.transformation
    
    scan_transformed = scan_downsampled.transform(reg.transformation)
    map_cloud += scan_transformed
    
    current_voxel = voxel_size
    map_cloud = map_cloud.voxel_down_sample(current_voxel)
    
    while len(map_cloud.points) > max_map_points:
        current_voxel *= 1.2
        map_cloud = map_cloud.voxel_down_sample(current_voxel)
    
    return global_pose, map_cloud

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

def run_icp_pipeline(data_file, max_scans=1000, voxel_size=0.01, max_map_points=3000, pose_sample_rate=10, progress_interval=50):
    global_pose = np.eye(4)
    map_cloud = o3d.geometry.PointCloud()
    pose_trajectory = []
    
    with open(data_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= max_scans:
                break

            try:
                scan_pcd, timestamp, points = line_to_pcd(line)
            except Exception as e:
                print(f"Line {i+1}: Parse error - {e}")
                skipped_scan_count += 1
                continue

            if not is_valid_scan(points):
                skipped_scan_count += 1
                continue

            global_pose, map_cloud, success = process_scan(
                scan_pcd, global_pose, map_cloud, voxel_size, max_map_points)

            if not success:
                failed_registration_count += 1

            valid_scan_count += 1

            if valid_scan_count % pose_sample_rate == 0:
                pose_trajectory.append(global_pose[:3, 3].copy())

            if valid_scan_count % progress_interval == 0:
                distance_traveled = np.linalg.norm(global_pose[:3, 3])
                success_rate = (valid_scan_count - failed_registration_count) / valid_scan_count * 100
                print(f"📊 Scan {valid_scan_count} (line {i+1})")
                print(f"   Timestamp: {timestamp}")
                print(f"   Map size: {len(map_cloud.points)} points")
                print(f"   Position: [{global_pose[0,3]:.3f}, {global_pose[1,3]:.3f}, {global_pose[2,3]:.3f}]")
                print(f"   Distance traveled: {distance_traveled:.3f}m")
                print(f"   Success rate: {success_rate:.1f}%")
                print()

    print("=" * 60)
    print("🎯 FINAL RESULTS")
    print("=" * 60)
    print(f"Valid scans processed: {valid_scan_count}")
    print(f"Scans skipped (invalid): {skipped_scan_count}")
    print(f"Failed registrations: {failed_registration_count}")

    success_rate = (valid_scan_count - failed_registration_count) / valid_scan_count * 100 if valid_scan_count > 0 else 0
    print(f"Success rate: {success_rate:.1f}%")

    print(f"Total lines read: {i + 1}")
    print(f"Final map size: {len(map_cloud.points)} points")

    print("\nFinal pose:")
    print(global_pose)

    total_distance = np.linalg.norm(global_pose[:3, 3])
    print(f"Total distance traveled: {total_distance:.3f}m")

    o3d.io.write_point_cloud("output_map.pcd", map_cloud)
    print(f"\n💾 Map saved to: output_map.pcd")

    if len(pose_trajectory) > 1:
        visualize_trajectory_and_map(pose_trajectory, map_cloud, valid_scan_count, failed_registration_count)
    else:
        print("⚠️  Warning: Not enough valid poses to visualize trajectory")

    return global_pose, map_cloud, pose_trajectory

if __name__ == "__main__":
    data_file = "data/data_recording_20251024_175931_lidar_reader__Decoded_VLP_to_PointCloud_1_1_XYZ_points.csv"
    max_scans = 100

    global_pose, map_cloud, pose_trajectory = run_icp_pipeline(
        data_file=data_file,
        max_scans=max_scans,
        voxel_size=0.02,
        max_map_points=5000,
        pose_sample_rate=10,
        progress_interval=50
    )
                
            scan_pcd, timestamp = line_to_pcd(line)
            global_pose, map_cloud = process_scan(scan_pcd, global_pose, map_cloud, voxel_size, max_map_points)
            
            if i % pose_sample_rate == 0:
                pose_trajectory.append(global_pose[:3, 3].copy())
            
            if (i + 1) % progress_interval == 0:
                print(f"Scan {i+1}: timestamp {timestamp}")
                print(f"Map size: {len(map_cloud.points)} points")
                print(f"Pose translation: [{global_pose[0,3]:.3f}, {global_pose[1,3]:.3f}, {global_pose[2,3]:.3f}]")
                rotation = np.linalg.norm(global_pose[:3, :3] - np.eye(3))
                print(f"Rotation magnitude: {rotation:.3f}\n")
    
    num_scans = i + 1
    print(f"\nFinal results after {num_scans} scans:")
    print(f"Final map size: {len(map_cloud.points)} points")
    print(f"Final pose:\n{global_pose}")
    
    o3d.io.write_point_cloud("output_map.pcd", map_cloud)
    
    visualize_trajectory_and_map(pose_trajectory, map_cloud, num_scans)
    
    return global_pose, map_cloud, pose_trajectory

if __name__ == "__main__":
    data_file = "data/lidardata.csv"
    max_scans = 5000
    
    global_pose, map_cloud, pose_trajectory = run_icp_pipeline(
        data_file=data_file,
        max_scans=max_scans,
        voxel_size=0.001,
        max_map_points=3000,
        pose_sample_rate=10,
        progress_interval=50
    )

