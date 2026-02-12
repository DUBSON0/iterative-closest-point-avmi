"""
Feature-based pre-alignment for 2D point clouds.

Pipeline
--------
1. Voxel downsample (coarse) for speed
2. PCA-based curvature  →  keypoint extraction (corners / endpoints)
3. Sorted-distance descriptors (rotation-invariant, no PCA-flip issues)
4. Nearest-neighbour matching with Lowe's ratio test
5. RANSAC rigid-transform estimation

The result is fed as an initial guess to ICP so it can handle large
rotations that would otherwise trap vanilla ICP in a local minimum.
"""

import numpy as np
from scipy.spatial import KDTree


# ── helpers ───────────────────────────────────────────────────────────────────

def _pairwise_sq(a, b):
    """Squared L2 distances between every row of *a* (N,D) and *b* (M,D).

    Returns an (N, M) matrix.  Uses the expansion ||a-b||² = ||a||² + ||b||² - 2 a·b
    which is much faster than broadcasting for large arrays.
    """
    a_sq = np.sum(a ** 2, axis=1, keepdims=True)   # (N, 1)
    b_sq = np.sum(b ** 2, axis=1, keepdims=True)   # (M, 1)
    return np.maximum(a_sq + b_sq.T - 2.0 * a @ b.T, 0.0)


# ── curvature & keypoints ────────────────────────────────────────────────────

def compute_curvature(points, k=10):
    """Estimate curvature at every point via PCA of *k*-nearest neighbours.

    Uses KDTree for O(N k log N) neighbour lookup instead of O(N²).
    Returns an array of curvature values in [0, 1].
    Higher value ≈ more corner-like (eigenvalues similar in magnitude).
    """
    n = len(points)
    k = min(k, n - 1)
    tree = KDTree(points)
    _, nn_all = tree.query(points, k=k + 1)             # (N, k+1) includes self
    curvatures = np.zeros(n)
    for i in range(n):
        nbrs = points[nn_all[i]]
        if len(nbrs) < 3:
            continue
        cov = np.cov(nbrs.T)
        ev = np.linalg.eigvalsh(cov)                    # ascending order
        curvatures[i] = ev[0] / (ev[-1] + 1e-10)       # ≈1 for corners, ≈0 for edges
    return curvatures


def extract_keypoints(points, curvatures, top_n=100, min_dist=0.3):
    """Select the *top_n* highest-curvature points with spatial non-max suppression."""
    order = np.argsort(-curvatures)                      # descending
    kp_idx = []
    kp_pts = []
    for idx in order:
        if len(kp_idx) >= top_n:
            break
        p = points[idx]
        if kp_pts:
            if np.min(np.linalg.norm(np.array(kp_pts) - p, axis=1)) < min_dist:
                continue
        kp_idx.append(idx)
        kp_pts.append(p)
    return np.array(kp_idx, dtype=int)


# ── descriptors ───────────────────────────────────────────────────────────────

def compute_descriptors(points, kp_idx, k=30):
    """Sorted-distance descriptor (rotation-invariant).

    For each keypoint, the descriptor is the sorted vector of Euclidean distances
    to the *k* nearest points.  Uses KDTree for O(n_kp * k * log N).
    """
    k = min(k, len(points) - 1)
    kp_pts = points[kp_idx]
    tree = KDTree(points)
    dists, _ = tree.query(kp_pts, k=k + 1)               # (n_kp, k+1) includes self
    descs = dists[:, 1:]                                  # skip self (column 0)
    return descs


# ── matching ──────────────────────────────────────────────────────────────────

def match_descriptors(da, db, ratio=0.8):
    """Nearest-neighbour matching with Lowe's ratio test.

    Returns a list of ``(idx_a, idx_b)`` pairs.
    """
    if len(da) == 0 or len(db) < 2:
        return []
    D = _pairwise_sq(da, db)                              # squared descriptor distances
    ratio_sq = ratio ** 2
    matches = []
    for i in range(len(da)):
        js = np.argsort(D[i])
        if D[i, js[0]] < ratio_sq * D[i, js[1]]:
            matches.append((i, js[0]))
    return matches


# ── RANSAC ────────────────────────────────────────────────────────────────────

def _rigid_from_points(src, dst):
    """Closed-form rigid (R, t) aligning src → dst.  Both (N, 2), N ≥ 2."""
    mu_s = src.mean(0)
    mu_d = dst.mean(0)
    W = (src - mu_s).T @ (dst - mu_d)
    U, _, Vt = np.linalg.svd(W)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mu_d - R @ mu_s
    return R, t


def ransac_align(kp_s, kp_t, matches, n_iter=1000, inlier_thresh=0.5):
    """RANSAC rigid 2-D transform from matched keypoints.

    Returns ``(R, t, n_inliers)`` or ``(None, None, 0)`` on failure.
    """
    if len(matches) < 2:
        return None, None, 0

    src = np.array([kp_s[m[0]] for m in matches])
    dst = np.array([kp_t[m[1]] for m in matches])
    n = len(matches)

    best_inliers = 0
    best_R, best_t = np.eye(2), np.zeros(2)

    for _ in range(n_iter):
        idx = np.random.choice(n, 2, replace=False)
        try:
            R, t = _rigid_from_points(src[idx], dst[idx])
        except Exception:
            continue
        err = np.linalg.norm((src @ R.T + t) - dst, axis=1)
        inliers = int(np.sum(err < inlier_thresh))
        if inliers > best_inliers:
            best_inliers = inliers
            best_R, best_t = R, t

    # Refine on all inliers of the best model
    if best_inliers >= 2:
        err = np.linalg.norm((src @ best_R.T + best_t) - dst, axis=1)
        mask = err < inlier_thresh
        if mask.sum() >= 2:
            best_R, best_t = _rigid_from_points(src[mask], dst[mask])
            best_inliers = int(mask.sum())

    return best_R, best_t, best_inliers


# ── correlative rotation search ───────────────────────────────────────────────

def rotation_search(
    source,
    target,
    voxel_size=0.3,
    angle_step_coarse=2.0,
    angle_step_fine=0.2,
):
    """Brute-force rotation search (correlative scan matching).

    Tries every rotation angle in [-180°, 180°) and picks the one that
    minimises mean nearest-neighbour distance after centroid alignment.
    Then refines with a finer angle sweep around the winner.

    This is extremely robust against large rotations because it evaluates
    alignment quality *directly* — no feature matching, no local minima.

    Parameters
    ----------
    source, target : (N, 2) arrays
    voxel_size : float
        Coarse voxel size used for speed (does not affect ICP voxel size).
    angle_step_coarse : float
        Degrees per step in the first sweep.
    angle_step_fine : float
        Degrees per step in the refinement sweep (± one coarse step
        around the coarse winner).

    Returns
    -------
    R : (2, 2)  rotation matrix
    t : (2,)    translation vector
    score : float  mean NN squared distance at the best angle
    """
    from .icp import voxel_downsample

    src = voxel_downsample(source, voxel_size)
    tgt = voxel_downsample(target, voxel_size)

    if len(src) < 5 or len(tgt) < 5:
        return np.eye(2), np.zeros(2), float("inf")

    mu_s = src.mean(axis=0)
    mu_t = tgt.mean(axis=0)
    src_c = src - mu_s                                    # centred source

    # Build KDTree on target once — each angle evaluation is O(N log M)
    tgt_tree = KDTree(tgt)

    def _score(angle_rad):
        ca, sa = np.cos(angle_rad), np.sin(angle_rad)
        R = np.array([[ca, -sa], [sa, ca]])
        rotated = src_c @ R.T + mu_t                     # rotate then shift to target centroid
        dists, _ = tgt_tree.query(rotated)
        return np.mean(dists ** 2)

    # ── coarse sweep ──────────────────────────────────────────────────────
    angles_coarse = np.deg2rad(np.arange(-180, 180, angle_step_coarse))
    scores_coarse = np.array([_score(a) for a in angles_coarse])
    best_idx = int(np.argmin(scores_coarse))
    best_angle = angles_coarse[best_idx]

    # ── fine sweep around the coarse winner ───────────────────────────────
    lo = best_angle - np.deg2rad(angle_step_coarse)
    hi = best_angle + np.deg2rad(angle_step_coarse)
    angles_fine = np.arange(lo, hi, np.deg2rad(angle_step_fine))
    scores_fine = np.array([_score(a) for a in angles_fine])
    best_idx_f = int(np.argmin(scores_fine))
    best_angle = angles_fine[best_idx_f]
    best_score = scores_fine[best_idx_f]

    # ── build output transform ────────────────────────────────────────────
    ca, sa = np.cos(best_angle), np.sin(best_angle)
    R = np.array([[ca, -sa], [sa, ca]])
    t = mu_t - R @ mu_s

    print(f"  Rotation search: best angle {np.degrees(best_angle):.1f}°, "
          f"score {best_score:.4f}")
    return R, t, best_score


# ── public entry points ───────────────────────────────────────────────────────

def feature_based_alignment(
    source,
    target,
    voxel_size=0.2,
    k_curvature=10,
    top_n=100,
    min_kp_dist=0.3,
    k_descriptor=30,
    ratio_threshold=0.8,
    ransac_iterations=1000,
    inlier_threshold=0.5,
):
    """Full feature-based alignment pipeline.

    Parameters
    ----------
    source, target : (N, 2) arrays in the same local frame convention used by
        the caller (sensor-local coordinates).
    voxel_size : float
        Coarse voxel size used *only* for the feature pipeline (does not affect
        the ICP voxel size).

    Returns
    -------
    R : (2, 2)  rotation matrix
    t : (2,)    translation vector
    n_inliers : int   number of RANSAC inliers (0 → alignment failed, caller
        should fall back to identity)
    """
    from .icp import voxel_downsample                     # avoid circular import

    src = voxel_downsample(source, voxel_size)
    tgt = voxel_downsample(target, voxel_size)

    if len(src) < 10 or len(tgt) < 10:
        return np.eye(2), np.zeros(2), 0

    # 1. Keypoints
    curv_s = compute_curvature(src, k=k_curvature)
    curv_t = compute_curvature(tgt, k=k_curvature)
    kpi_s = extract_keypoints(src, curv_s, top_n=top_n, min_dist=min_kp_dist)
    kpi_t = extract_keypoints(tgt, curv_t, top_n=top_n, min_dist=min_kp_dist)

    if len(kpi_s) < 2 or len(kpi_t) < 2:
        return np.eye(2), np.zeros(2), 0

    # 2. Descriptors
    desc_s = compute_descriptors(src, kpi_s, k=k_descriptor)
    desc_t = compute_descriptors(tgt, kpi_t, k=k_descriptor)

    # 3. Matching
    matches = match_descriptors(desc_s, desc_t, ratio=ratio_threshold)
    if len(matches) < 2:
        return np.eye(2), np.zeros(2), 0

    # 4. RANSAC
    R, t, n_inliers = ransac_align(
        src[kpi_s],
        tgt[kpi_t],
        matches,
        n_iter=ransac_iterations,
        inlier_thresh=inlier_threshold,
    )

    if R is None:
        return np.eye(2), np.zeros(2), 0

    print(f"  Feature alignment: {len(matches)} matches, {n_inliers} inliers")
    return R, t, n_inliers
