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

def find_nearest_neighbor_indices(source, target):
    """Return the index of the nearest target point for every source point."""
    distances = np.sum((source[:, np.newaxis, :] - target[np.newaxis, :, :]) ** 2, axis=2)
    return np.argmin(distances, axis=1)


# ── Point-to-line helpers (2-D only) ─────────────────────────────────────────

def estimate_normals_2d(points, k=10):
    """Estimate 2-D surface normals via PCA of k-nearest neighbours.

    The normal at each point is the eigenvector corresponding to the
    *smallest* eigenvalue of the local covariance — i.e. perpendicular
    to the dominant surface direction.

    Returns an (N, 2) array of unit normals.
    """
    n = len(points)
    k = min(k, n - 1)
    normals = np.zeros_like(points)
    for i in range(n):
        dists_sq = np.sum((points - points[i]) ** 2, axis=1)
        nn_idx = np.argpartition(dists_sq, k + 1)[: k + 1]
        nbrs = points[nn_idx]
        cov = np.cov(nbrs.T)
        eigvals, eigvecs = np.linalg.eigh(cov)          # ascending eigenvalues
        normals[i] = eigvecs[:, 0]                       # smallest → normal
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(norms, 1e-10)
    return normals


def _point_to_line_solve_2d(source_pts, target_pts, target_normals, nn_indices):
    """One point-to-line ICP step via linearised least-squares.

    Minimises  Σ (n_i · (R(θ) p_i + t − q_i))²  using a small-angle
    approximation  cos θ ≈ 1, sin θ ≈ θ.  This turns the problem into a
    linear system in (θ, tx, ty) that is solved in closed form.

    Returns R (2×2), t (2,).
    """
    q  = target_pts[nn_indices]             # matched target points
    nm = target_normals[nn_indices]          # normals at matched targets
    p  = source_pts

    nx, ny = nm[:, 0], nm[:, 1]
    px, py = p[:, 0],  p[:, 1]
    dx, dy = px - q[:, 0], py - q[:, 1]

    # Coefficient of θ for each correspondence
    c = ny * px - nx * py

    # Linear system  A x = b,  x = [θ, tx, ty]
    A = np.column_stack([c, nx, ny])
    b = -(nx * dx + ny * dy)

    ATA = A.T @ A
    ATb = A.T @ b
    try:
        x = np.linalg.solve(ATA, ATb)
    except np.linalg.LinAlgError:
        return np.eye(2), np.zeros(2)

    theta, tx, ty = x
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t,  cos_t]])
    t = np.array([tx, ty])
    return R, t

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


def ICP(source, target, error_threshold, max_iterations, voxel_size,
        R_init=None, t_init=None, method="point_to_point", normal_k=10):
    """Iterative Closest Point.

    Parameters
    ----------
    method : str
        ``"point_to_point"`` — classic SVD-based ICP.
        ``"point_to_line"``  — minimises distance to local surface tangent
        (2-D only; falls back to point-to-point for 3-D data).
    normal_k : int
        Neighbours used when estimating target normals (point-to-line only).
    """
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

    use_p2l = (method == "point_to_line" and source.shape[1] == 2)

    # Pre-compute target normals once (they don't change across iterations)
    target_normals = None
    if use_p2l:
        target_normals = estimate_normals_2d(target, k=normal_k)

    prev_error = float('inf')
    for iteration in range(max_iterations):
        # ── find correspondences ──────────────────────────────────────
        nn_indices = find_nearest_neighbor_indices(transformed, target)
        nearest = target[nn_indices]

        # ── solve for incremental (r, t) ─────────────────────────────
        if use_p2l:
            r, t = _point_to_line_solve_2d(
                transformed, target, target_normals, nn_indices,
            )
        else:
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

        # ── accumulate & apply ────────────────────────────────────────
        r_total = np.dot(r, r_total)
        t_total = np.dot(t_total, r.T) + t
        transformed = np.dot(transformed, r.T) + t

        # ── convergence check (always point-to-point error) ──────────
        error = np.mean(np.sum((nearest - transformed) ** 2, axis=1))
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


