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

def line_to_pcd(line):
    timestamp, points = parse_line_lidar_data(line)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd, timestamp

def process_scan(scan_pcd, global_pose, map_cloud, voxel_size=0.01, max_map_points=3000):
    scan_downsampled = scan_pcd.voxel_down_sample(voxel_size)
    
    if len(map_cloud.points) == 0:
        map_cloud += scan_downsampled
        return global_pose, map_cloud
    
    reg = o3d.pipelines.registration.registration_icp(
        scan_downsampled, map_cloud, 0.05,
        init=global_pose,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria=o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=10000,
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

<<<<<<< HEAD
voxel_size = 0.05
source_down = source.voxel_down_sample(voxel_size)
target_down = target.voxel_down_sample(voxel_size)

radius_normal = voxel_size * 2
source_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
target_down.estimate_normals(
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

radius_feature = voxel_size * 5
source_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    source_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
target_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
    target_down,
    o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

distance_threshold = voxel_size * 1.5
ransac_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
    source_down, target_down, source_fpfh, target_fpfh,
    mutual_filter=True,
    max_correspondence_distance=distance_threshold,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
    ransac_n=3,
    checkers=[
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
        o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
    ],
    criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999)
)
print("RANSAC transformation:")
print(ransac_result.transformation)

icp_result = o3d.pipelines.registration.registration_icp(
    source_down, target_down, 0.1, ransac_result.transformation,
    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    criteria=o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
)
print("\nRefined ICP transformation:")
print(icp_result.transformation)
=======
global_pose = np.eye(4)
map_cloud = o3d.geometry.PointCloud()
pose_trajectory = []

data_file = "data/lidardata.csv"
max_scans = 2000
pose_sample_rate = 10

with open(data_file, 'r') as f:
    for i, line in enumerate(f):
        if i >= max_scans:
            break
            
        scan_pcd, timestamp = line_to_pcd(line)
        global_pose, map_cloud = process_scan(scan_pcd, global_pose, map_cloud)
        
        if i % pose_sample_rate == 0:
            pose_trajectory.append(global_pose[:3, 3].copy())
        
        if (i + 1) % 50 == 0:
            print(f"Scan {i+1}: timestamp {timestamp}")
            print(f"Map size: {len(map_cloud.points)} points")
            print(f"Pose translation: [{global_pose[0,3]:.3f}, {global_pose[1,3]:.3f}, {global_pose[2,3]:.3f}]")
            rotation = np.linalg.norm(global_pose[:3, :3] - np.eye(3))
            print(f"Rotation magnitude: {rotation:.3f}\n")

print(f"\nFinal results after {i+1} scans:")
print(f"Final map size: {len(map_cloud.points)} points")
print(f"Final pose:\n{global_pose}")
o3d.io.write_point_cloud("output_map.pcd", map_cloud)
print("Map saved to output_map.pcd")

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
ax.set_title(f'Pose Trajectory Over Map ({len(map_points)} points)')
ax.legend()

plt.tight_layout()
plt.savefig('trajectory_and_map_3d.png', dpi=150)
print("3D plot saved to trajectory_and_map_3d.png")

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)

ax2.scatter(map_points[:, 0], map_points[:, 1], c='gray', s=1, alpha=0.3, label='Map')
ax2.plot(pose_trajectory[:, 0], pose_trajectory[:, 1], 'r-', linewidth=3, label='Trajectory')
ax2.scatter(pose_trajectory[0, 0], pose_trajectory[0, 1], c='g', s=150, marker='o', label='Start', edgecolors='black', linewidth=2)
ax2.scatter(pose_trajectory[-1, 0], pose_trajectory[-1, 1], c='b', s=150, marker='s', label='End', edgecolors='black', linewidth=2)

ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_title(f'Pose Trajectory Over Map - XY Plane ({len(map_points)} points)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.axis('equal')

plt.tight_layout()
plt.savefig('trajectory_and_map_2d.png', dpi=150)
print("2D plot saved to trajectory_and_map_2d.png")
plt.show()
>>>>>>> 529bf5c (The pose estimation is way off. A couple of good results but not great.)

