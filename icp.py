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
    def plot_3d_voxel_map(self):
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
    def plot_2d_map(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(self.map[:, 0], self.map[:, 1], c='tab:blue', s=2, label='2D Map')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title('2D Point Map (XY Projection)')
        ax.legend()
        plt.axis('equal')
        plt.show()

class OccupancyGrid:
    def __init__(self, map_resolution_m, map_size_m):
        self.lidar_z_position = 1
        self.scan_vertical_range = 1
        self.map_resolution_m = map_resolution_m
        self.map_size_m = map_size_m
        self.map_ratio = int(self.map_size_m/self.map_resolution_m)
        self.grid = np.zeros((self.map_ratio, self.map_ratio))
        self.grid[:] = 0.5
        self.grid_origin = np.floor(self.map_ratio/2).astype(int)
    def update(self, points, pose):
        # Use Bayes Filter
        floor_points = np.floor(points/self.map_resolution_m).astype(int) # This is to normalize the value of the lidar points to the size of the grid cells.
        for point in floor_points:
            if point[2] > ((self.lidar_z_position-self.scan_vertical_range)/self.map_resolution_m) and point[2] < ((self.lidar_z_position+self.scan_vertical_range)/self.map_resolution_m):
                self.grid[point[0]+self.grid_origin][point[1]+self.grid_origin] = 1
        return self.grid
    def plot_grid(self):
        fig, ax = plt.subplots(figsize=(8, 8))
        n = self.grid.shape[0]
        extent = [0, n, 0, n]
        show_grid = np.zeros_like(self.grid)
        show_grid[self.grid == 1] = 1
        cmap = plt.cm.Blues
        ax.imshow(show_grid.T, origin='lower', extent=extent, cmap=cmap, vmin=0, vmax=1, interpolation='none')
        ax.set_xticks(np.arange(0, n+1, 1))
        ax.set_yticks(np.arange(0, n+1, 1))
        ax.grid(which='both', color='gray', linestyle='-', linewidth=0.7)
        ax.set_xlim(0, n)
        ax.set_ylim(0, n)
        ax.set_xlabel('X (Grid index)')
        ax.set_ylabel('Y (Grid index)')
        ax.set_title('Occupancy Grid')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        plt.tight_layout()
        plt.show()
    def write_grid_to_file(self, file_path):
        with open(file_path, 'w') as file:
            for x in range(self.grid.shape[0]):
                for y in range(self.grid.shape[1]):
                    file.write(str(self.grid[x][y]) + " ")
                file.write("\n")
            file.write("\n")

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


def ICP(source, target, error_threshold, max_iterations, voxel_size):
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
        source_centered = transformed - mu_source
        target_centered = nearest - mu_target
        w = np.dot(source_centered.T, target_centered)
        u, s, vt = np.linalg.svd(w)
        r = np.dot(vt.T, u.T)
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = np.dot(u, vt)
        t = mu_target - np.dot(r, mu_source)
        r_total = np.dot(r, r_total)
        t_total = np.dot(t_total, r.T)+t
        transformed = np.dot(transformed, r.T) + t
        error = np.mean(np.sum((nearest-transformed) ** 2, axis=1))
        if abs(prev_error-error) < error_threshold:
            return r, t, error
        prev_error = error
    return r_total, t_total, error

def run_icp(data_file, num_scans=10, map=None, occupancy_grid=None):
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
            r, t, error = ICP(prev_points, points, voxel_size=0.6, error_threshold=0.001, max_iterations=10)
            transform = np.eye(4)
            transform[:3, :3] = r
            transform[:3, 3] = t
            global_pose = np.dot(global_pose, transform)
            pose_trajectory.append(global_pose.copy())
            prev_points = points
            scans_processed += 1
            occupancy_grid.update(np.dot(points, r.T) + t, global_pose)
            if map:
                map.update(points)
            if scans_processed >= num_scans:
                break
            print("Scan: ", scans_processed, "Error: ", error)
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
    map = Map(voxel_size=0.003)
    occupancy_grid = OccupancyGrid(map_resolution_m=0.5, map_size_m=50)
    time, _, pose_trajectory = run_icp('data/lidardata.csv', map=map, num_scans=5000, occupancy_grid=occupancy_grid)
    occupancy_grid.write_grid_to_file('data/occupancy_grid.txt')
    occupancy_grid.plot_grid()
    plot_pose_trajectory(pose_trajectory)
    print("Time start: ", str(int(time[0])/1e6), "Time end: ", str(int(time[-1])/1e6))
    map.plot_2d_map()
if __name__ == '__main__':
    main()
