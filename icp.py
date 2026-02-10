import numpy as np
import pcview

def open_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

def parse_line_lidar_data(data):
    elements = data.strip().replace(";", " ").split()
    if len(elements) < 2:
        raise ValueError("...")

    timestamp = elements[0]
    float_values = [float(v) for v in elements[1:]]
    if len(float_values) % 3 != 0:
        raise ValueError("")
    points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])
    mask = np.all(points == 0, axis =1)
    points = points[~mask]
    return timestamp, points

def parse_line_teapot_data(data):
    elements = data.strip().replace(",", " ").replace("\n", " ").split()
    print("len: ", len(elements))
    float_values = [float(v) for v in elements]
    points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])
    print("len: ", len(points))
    return points


def center_of_mass(points):
    return np.array(np.mean(points, axis=0))

def find_nearest_neighbors(source, target):
    distances = np.sum((source[:, np.newaxis, :] - target[np.newaxis, :, :]) ** 2, axis=2)
    nearest_indices = np.argmin(distances, axis=1) # The indices of the points in source corresdpond to the indices here in the target.
    return target[nearest_indices] 

def voxel_downsample(points, voxel_size):
    dim = points.shape[1]
    min_bound = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    unique_indices, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    downsampled = np.zeros((len(unique_indices), dim))
    counts = np.zeros(len(unique_indices))
    for i in range(len(points)):
        idx = inv[i]
        downsampled[idx] += points[i]
        counts[idx] += 1
    downsampled /= counts[:, np.newaxis]
    return downsampled


def ICP(source, target, error_threshold, max_iterations, voxel_size, R_init=None, t_init=None):
    source = voxel_downsample(source, voxel_size)
    target = voxel_downsample(target, voxel_size)

    if R_init is not None and t_init is not None:
        transformed = source @ R_init.T + t_init
        r_total = R_init.copy()
        t_total = t_init.copy()
    else:
        transformed = source.copy()
        r_total = np.eye(source.shape[1])
        t_total = np.zeros(source.shape[1])

    prev_error = float('inf')
    # X is target, P is source
    for iteration in range(max_iterations):
        nearest = find_nearest_neighbors(transformed, target)
        mu_source = center_of_mass(transformed)
        mu_target = center_of_mass(nearest)
        source_centered = transformed - mu_source
        target_centered = nearest - mu_target
        w = np.dot(source_centered.T, target_centered)
        u, s, vt = np.linalg.svd(w)
        r = np.dot(vt.T, u.T)
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = np.dot(vt.T, u.T)
        t = mu_target - np.dot(r, mu_source)
        r_total = np.dot(r, r_total)
        t_total = np.dot(t_total, r.T)+t
        transformed = np.dot(transformed, r.T) + t
        error = np.mean(np.sum((nearest-transformed) ** 2, axis=1))
        delta = abs(prev_error - error)
        if delta < error_threshold:
            print(f"  ICP converged: iter={iteration}, error={error:.8f}, delta={delta:.2e}")
            return r_total, t_total, error
        prev_error = error
    print(f"  ICP max iterations reached: iter={max_iterations}, error={error:.8f}")
    return r_total, t_total, error

def run_icp(scan_stream, num_scans=None, error_threshold=1e-5, max_iterations=100, voxel_size=0.5):
    global_pose = np.eye(4)
    pose_trajectory = []
    prev_points = None
    scans_processed = 0
    for timestamp, points in scan_stream:
        if prev_points is None:
            prev_points = points
            continue
        r, t, error = ICP(
            prev_points,
            points,
            error_threshold=error_threshold,
            max_iterations=max_iterations,
            voxel_size=voxel_size,
        )
        R_new = global_pose[:3, :3] @ r.T
        global_pose[:3, :3] = R_new
        global_pose[:3, 3] = global_pose[:3, 3] - R_new @ t
        pose_trajectory.append(global_pose.copy())
        prev_points = points
        scans_processed += 1
        if num_scans is not None and scans_processed >= num_scans:
            break
        print("Scan: ", scans_processed, "Error: ", error)
    return global_pose, pose_trajectory

def run_dual_file_icp():
    file_path_a = 'data/cow.csv'
    file_path_b = 'data/cowt.csv'
    line = open_file(file_path_a)
    points_a = parse_line_teapot_data(line)
    line = open_file(file_path_b)
    points_b = parse_line_teapot_data(line)
    r, t, error = ICP(points_a, points_b, voxel_size=0.01, error_threshold=0.00001, max_iterations=1000)
    print("r: ", r)
    print("t: ", t)
    print("error: ", error)
    visualize_transformation(r, t, points_a, points_b)

def visualize_transformation(r, t, source, target):
    transformed_source = np.dot(source, r.T) + t
    pcview.visualize_point_clouds(
        [source, transformed_source, target],
        labels=["Source", "Transformed Source", "Target"],
        colors=["green", "blue", "orange"],
        point_size=5,
        background_color="black",
        window_size=(900, 700),
        show_legend=True,
        enable_toggles=True,
    )


