import numpy as np

def open_file(file_path):
    with open(file_path, 'r') as file:
        return file.readline()

def parse_line_lidar_data(data):
    elements = data.strip().replace(";", " ").split()
    if len(elements) < 2:
        raise ValueError("...")

    timestamp = elements[0]
    float_values = [float(v) for v in elements[1:]]
    if len(float_values) % 3 != 0:
        raise ValueError("")
    points = np.array([float_values[i:i+3] for i in range(0, len(float_values), 3)])

    return timestamp, points

def center_of_mass(points):
    return np.array(np.mean(points, axis=0))

def find_nearest_neighbors(source, target):
    distances = np.sum((source[:, np.newaxis, :] - target[np.newaxis, :, :]) ** 2, axis=2)
    nearest_indices = np.argmin(distances, axis=1)
    return target[nearest_indices]

def ICP(source, target, error_threshold=0.01, max_iterations=100):
    transformed = source.copy()
    prev_error = float('inf')
    
    for iteration in range(max_iterations):
        nearest = find_nearest_neighbors(transformed, target)
        
        mu_source = center_of_mass(transformed)
        mu_target = center_of_mass(nearest)
        
        source_centered = transformed - mu_source
        target_centered = nearest - mu_target
        
        w = np.dot(target_centered.T, source_centered)
        u, s, vt = np.linalg.svd(w)
        r = np.dot(u, vt)
        
        if np.linalg.det(r) < 0:
            vt[-1, :] *= -1
            r = np.dot(u, vt)
        
        t = mu_target - np.dot(r, mu_source)
        
        transformed = np.dot(transformed, r.T) + t
        
        error = np.mean(np.sum((transformed - nearest) ** 2, axis=1))
        print(f"Iteration {iteration + 1}: error = {error:.6f}")
        
        if abs(prev_error - error) < error_threshold:
            print(f"Converged after {iteration + 1} iterations")
            return r, t, error
        
        prev_error = error
    
    print(f"Max iterations ({max_iterations}) reached")
    return r, t, error

def main():
    file_path_a = 'data/real_lidar1.csv'
    file_path_b = 'data/real_lidar2.csv'
    line = open_file(file_path_a)
    timestamp, points_a = parse_line_lidar_data(line)
    line = open_file(file_path_b)
    timestamp, points_b = parse_line_lidar_data(line)
    r, t, error = ICP(points_a, points_b)
    print("Rotation matrix: ", r)

if __name__ == '__main__':
    main()
