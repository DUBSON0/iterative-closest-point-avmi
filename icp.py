import numpy as np
import matplotlib.pyplot as plt
import datetime

class Map:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.map = None

    def update(self, points):
        if self.map is None:
            self.map = points
        else:
            self.map = np.concatenate([self.map, points], axis=0)
        if self.map.shape[0] > 10000:
            self.map = voxel_downsample(self.map, 0.05)

    def plot_map(self):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        z = self.map[:, 2]
        norm = (z - z.min()) / (z.max() - z.min() + 1e-12)
        scatter = ax.scatter(self.map[:, 0], self.map[:, 1], self.map[:, 2], c=norm, cmap='coolwarm_r', s=5, label='Map')
        fig.colorbar(scatter, ax=ax, label='Z')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.show()
    def occupancy_grid(self):
        grid = np.zeros((100, 100, 100))
        for point in self.map:
            grid[int(point[0]), int(point[1]), int(point[2])] = 1
        return grid
class OccupancyGrid:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.grid = None
    def update(self, points):
        if self.grid is None:
            self.grid = np.zeros((100, 100, 100))
        for point in points:
            idx = (int(point[0]), int(point[1]), int(point[2]))
            current_prob = self.grid[idx]
            new_prob = current_prob + (1.0 - current_prob) * 0.5
            self.grid[idx] = new_prob

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
    #print("len: ", len(elements))
    float_values = [float(v) for v in elements]
    points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])
    #print("len: ", len(points))
    return points


def center_of_mass(points):
    return np.array(np.mean(points, axis=0))

def find_nearest_neighbors(source, target):
    distances = np.sum((source[:, np.newaxis, :] - target[np.newaxis, :, :]) ** 2, axis=2)
    nearest_indices = np.argmin(distances, axis=1) # The indices of the points in source corresdpond to the indices here in the target.
    return target[nearest_indices] 

def voxel_downsample(points, voxel_size):
    min_bound = np.min(points, axis=0)
    voxel_indices = np.floor((points - min_bound) / voxel_size).astype(int)
    unique_indices, inv = np.unique(voxel_indices, axis=0, return_inverse=True)
    downsampled = np.zeros((len(unique_indices), 3))
    counts = np.zeros(len(unique_indices))
    for i in range(len(points)):
        idx = inv[i]
        downsampled[idx] += points[i]
        counts[idx] += 1
    downsampled /= counts[:, np.newaxis]
    return downsampled


def ICP(source, target, error_threshold, max_iterations, voxel_size, map=None):
    source = voxel_downsample(source, voxel_size)
    target = voxel_downsample(target, voxel_size)
    transformed = source.copy()
    prev_error = float('inf')
    r_total = np.eye(source.shape[1])
    t_total = np.zeros(source.shape[1])
    #X is target, P is source 
    for iteration in range(max_iterations):
        nearest = find_nearest_neighbors(transformed, target)
        mu_source = center_of_mass(transformed)
        mu_target = center_of_mass(nearest)
        #print("mu_source: ", mu_source)
        #print("mu_target: ", mu_target)
        source_centered = transformed - mu_source
        target_centered = nearest - mu_target
        #print("shape of source_centered: ", source_centered.T.shape)
        #print("shape of target_centered: ", target_centered.shape)
        w = np.dot(source_centered.T, target_centered)
        #print("shape of w: ", w.shape)
        u, s, vt = np.linalg.svd(w)
        r = np.dot(vt.T, u.T)
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = np.dot(u, vt)
        t = mu_target - np.dot(r, mu_source)
        r_total = np.dot(r, r_total)
        t_total = np.dot(t_total, r.T)+t
        #print("shape of r: ", r.shape)
        #print("shape of t: ", t.shape)
        transformed = np.dot(transformed, r.T) + t
        #print("shape of transformed: ", transformed)
        #print("shape of transformed: ", transformed.shape)
        error = np.mean(np.sum((nearest-transformed) ** 2, axis=1))
        if abs(prev_error-error) < error_threshold:
            return r, t, error
        prev_error = error
        if map is not None:
            #print("Updating map")
            #print("shape of transformed: ", transformed.shape)
            map.update(np.dot(source, r.T) + t)
    return r_total, t_total, error

def run_icp(data_file, num_scans=10, map=None):
    global_pose = np.eye(4)
    pose_trajectory = []
    prev_points = None
    scans_processed = 0
    time_segment =[] 
    with open(data_file, 'r') as file:
        for line in file:
            timestamp, points = parse_line_lidar_data(line)
            time_segment.append(timestamp)
            if prev_points is None:
                prev_points = points
                continue
            r, t, error = ICP(prev_points, points, voxel_size=0.6, error_threshold=0.001, max_iterations=10, map=map)

            transform = np.eye(4)
            transform[:3, :3] = r
            transform[:3, 3] = t
            global_pose = np.dot(global_pose, transform)
            pose_trajectory.append(global_pose.copy())
            prev_points = points
            scans_processed += 1
            if scans_processed >= num_scans:
                break
            print("Scan: ", scans_processed, "Error: ", error)
            #print("Tranform:\n", transform, "\n")
    return time_segment, global_pose, pose_trajectory

def plot_pose_trajectory(pose_trajectory):
    if not pose_trajectory:
        return
    positions = np.array([pose[:3, 3] for pose in pose_trajectory])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], c='tab:blue')
    ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], c='tab:green', s=30)
    ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], c='tab:red', s=30)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Pose Trajectory')
    plt.show()

def run_dual_file_icp():
    file_path_a = 'data/outdoor-sample-1.csv'
    file_path_b = 'data/outdoor-sample-2.csv'
    line = open_file(file_path_a)
    timestampe, points_a = parse_line_lidar_data(line)
    print(points_a)
    line = open_file(file_path_b)
    timestamp, points_b = parse_line_lidar_data(line)
    r, t, error = ICP(points_a, points_b, voxel_size=0.05, error_threshold=0.001, max_iterations=10)
    #print("r: ", r)
    #print("t: ", t)
    #print("error: ", error)
    visualize_tranforation(r, t, points_a, points_b)

def visualize_tranforation(r, t, source, target):
    transformed_source = np.dot(source, r.T) + t
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(source[:, 0], source[:, 1], source[:, 2], c='tab:green', s=5, label='Source')
    ax.scatter(transformed_source[:, 0], transformed_source[:, 1], transformed_source[:, 2], c='tab:blue', s=5, label='Transformed Source')
    ax.scatter(target[:, 0], target[:, 1], target[:, 2], c='tab:orange', s=5, label='Target')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.show()


def main():
    #global_pose, pose_trajectory = run_icp(data_file='data/lidardata.csv', num_scans=500, voxel_size=0.3)
    #run_dual_file_icp()
    map = Map(voxel_size=0.003)
    time, _, pose_trajectory = run_icp('data/lidardata.csv', map=map, num_scans=1000)
    plot_pose_trajectory(pose_trajectory)
    print("Time start: ", str(int(time[0])/1e6), "Time end: ", str(int(time[-1])/1e6))
    map.plot_map()
if __name__ == '__main__':
    main()
