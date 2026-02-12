import argparse
import yaml
import numpy as np
import pyvista as pv

from services.lidar_service import LidarService
from services.imu_service import IMUService
from utilities.icp import ICP, voxel_downsample
from utilities.features import feature_based_alignment, rotation_search
from utilities.mapping import OccupancyGrid2D
from utilities.pose_graph import (
    PoseGraph2D,
    pose_matrix_to_vec,
    relative_transform_vec,
    pose_vec_to_matrix,
)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_points(points, z_min=0.2, z_max=2.0):
    """Keep only points with z in [z_min, z_max].

    Returns
    -------
    points_2d : ndarray, shape (M, 2)
        Filtered points projected to x, y (for ICP).
    points_3d : ndarray, shape (M, 3)
        Filtered points with original z preserved (for 2.5-D mapping).
    """
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    pts_3d = points[mask].copy()
    pts_2d = pts_3d[:, :2].copy()
    return pts_2d, pts_3d


def compute_bounds_from_scan(points_2d, margin=50.0):
    min_x = float(np.min(points_2d[:, 0]) - margin)
    max_x = float(np.max(points_2d[:, 0]) + margin)
    min_y = float(np.min(points_2d[:, 1]) - margin)
    max_y = float(np.max(points_2d[:, 1]) + margin)
    return min_x, max_x, min_y, max_y


def apply_incremental_pose_2d(global_pose, r, t):
    """Accumulate the inverse of ICP's 2D forward transform."""
    T_inv = np.eye(3)
    T_inv[:2, :2] = r.T
    T_inv[:2, 2] = -r.T @ t
    return global_pose @ T_inv


def transform_points_2d(points_2d, pose):
    """Transform 2D points using a 3x3 homogeneous pose."""
    R = pose[:2, :2]
    t = pose[:2, 2]
    return points_2d @ R.T + t


def transform_points_25d(points_3d, pose):
    """Apply a 2-D pose to x, y while preserving z (for 2.5-D mapping)."""
    R = pose[:2, :2]
    t = pose[:2, 2]
    result = points_3d.copy()
    result[:, :2] = points_3d[:, :2] @ R.T + t
    return result


def _run_icp_pair(source, target, icp_cfg, feat_cfg, alignment_method):
    """Run pre-alignment + ICP between two 2-D scans.

    Returns (R, t, error) — same as ICP().
    """
    R_init, t_init = None, None

    if alignment_method in ("rotation_search", "both"):
        R_init, t_init, _ = rotation_search(
            source, target,
            voxel_size=feat_cfg.get("rotation_voxel_size", 0.3),
            angle_step_coarse=feat_cfg.get("angle_step_coarse", 2.0),
            angle_step_fine=feat_cfg.get("angle_step_fine", 0.2),
        )

    if alignment_method in ("features", "both"):
        fa_src = source
        if R_init is not None:
            fa_src = source @ R_init.T + t_init
        R_feat, t_feat, n_inliers = feature_based_alignment(
            fa_src, target,
            voxel_size=feat_cfg.get("voxel_size", 0.2),
            k_curvature=feat_cfg.get("k_curvature", 10),
            top_n=feat_cfg.get("top_n", 100),
            min_kp_dist=feat_cfg.get("min_kp_dist", 0.3),
            k_descriptor=feat_cfg.get("k_descriptor", 30),
            ratio_threshold=feat_cfg.get("ratio_threshold", 0.8),
            ransac_iterations=feat_cfg.get("ransac_iterations", 1000),
            inlier_threshold=feat_cfg.get("inlier_threshold", 0.5),
        )
        if n_inliers >= feat_cfg.get("min_inliers", 3):
            if R_init is not None:
                R_init = R_feat @ R_init
                t_init = t_init @ R_feat.T + t_feat
            else:
                R_init, t_init = R_feat, t_feat

    return ICP(
        source, target,
        error_threshold=icp_cfg.get("error_threshold", 1e-7),
        max_iterations=icp_cfg.get("max_iterations", 100),
        voxel_size=icp_cfg.get("voxel_size", 0.06),
        R_init=R_init, t_init=t_init,
        method=icp_cfg.get("method", "point_to_line"),
        normal_k=icp_cfg.get("normal_k", 10),
    )


# ── Submap helpers ────────────────────────────────────────────────────────────

def _build_submap(buffer, voxel_sz):
    """Concatenate global-frame scan buffers and voxel-downsample."""
    if not buffer:
        return np.empty((0, 2))
    combined = np.vstack(buffer)
    return voxel_downsample(combined, voxel_sz)


def _submap_rotation_search(source_local, submap_global, predicted_pose,
                             angle_range=60.0, angle_step=2.0,
                             fine_step=0.5, voxel_size=0.3):
    """Search rotation around predicted pose for best submap alignment.

    Tries rotation offsets in [-angle_range, +angle_range] degrees.
    After finding the best rotation, **refines the translation** with a
    single nearest-neighbour centroid step so the returned (R, t) is no
    longer locked to the predicted translation.

    Uses KDTree for O(N log M) scoring per angle instead of O(N*M).
    """
    from scipy.spatial import KDTree

    src = voxel_downsample(source_local, voxel_size)
    tgt = voxel_downsample(submap_global, voxel_size)

    if len(src) < 5 or len(tgt) < 5:
        return predicted_pose[:2, :2], predicted_pose[:2, 2]

    pred_t = predicted_pose[:2, 2]
    pred_theta = np.arctan2(predicted_pose[1, 0], predicted_pose[0, 0])

    # Build KDTree on target ONCE — each angle evaluation is now
    # O(N log M) instead of O(N*M).
    tgt_tree = KDTree(tgt)

    def _score(theta):
        ca, sa = np.cos(theta), np.sin(theta)
        R = np.array([[ca, -sa], [sa, ca]])
        rotated = src @ R.T + pred_t                           # (N, 2)
        dists, _ = tgt_tree.query(rotated)
        return np.mean(dists ** 2)

    # ── coarse sweep ──────────────────────────────────────────────────
    offsets = np.deg2rad(np.arange(-angle_range, angle_range + angle_step,
                                    angle_step))
    angles = pred_theta + offsets
    scores = np.array([_score(a) for a in angles])
    best_idx = int(np.argmin(scores))
    best_angle = angles[best_idx]

    # ── fine sweep around coarse winner ───────────────────────────────
    fine_lo = best_angle - np.deg2rad(angle_step)
    fine_hi = best_angle + np.deg2rad(angle_step)
    fine_angles = np.arange(fine_lo, fine_hi, np.deg2rad(fine_step))
    if len(fine_angles) > 0:
        fine_scores = np.array([_score(a) for a in fine_angles])
        best_angle = fine_angles[int(np.argmin(fine_scores))]

    correction = np.degrees(best_angle - pred_theta)
    if abs(correction) > 1.0:
        print(f"  Submap rotation correction: {correction:+.1f}°")

    ca, sa = np.cos(best_angle), np.sin(best_angle)
    R_best = np.array([[ca, -sa], [sa, ca]])

    # ── refine translation with one NN centroid step ──────────────────
    rotated_src = src @ R_best.T
    placed = rotated_src + pred_t
    nn_dists, nn_idx = tgt_tree.query(placed)
    nn_dists_sq = nn_dists ** 2
    # Keep the closest 80 % to suppress outlier correspondences
    dist_thresh = np.percentile(nn_dists_sq, 80)
    inlier_mask = nn_dists_sq <= dist_thresh
    if inlier_mask.sum() >= 5:
        matched = tgt[nn_idx]
        refined_t = np.mean(matched[inlier_mask] - rotated_src[inlier_mask],
                            axis=0)
    else:
        refined_t = pred_t

    return R_best, refined_t


def _attempt_submap_icp(source, submap, predicted,
                         imu_yaw, imu_narrow,
                         sub_rot_range, sub_rot_step, sub_rot_fine,
                         sub_rot_voxel, icp_cfg, sub_corr_dist):
    """Run rotation-search + point-to-point ICP against *submap*.

    If *imu_yaw* is not None the predicted rotation is overridden with
    the IMU absolute yaw and only a narrow search (± *imu_narrow* °) is
    performed; otherwise the full rotation range is swept.

    Returns ``(r, t, error)`` — same convention as ``ICP()``.
    """
    pred = predicted.copy()

    if imu_yaw is not None:
        ca, sa = np.cos(imu_yaw), np.sin(imu_yaw)
        pred[:2, :2] = np.array([[ca, -sa], [sa, ca]])
        angle_range = imu_narrow
        angle_step  = 0.5
    else:
        angle_range = sub_rot_range
        angle_step  = sub_rot_step

    R_init, t_init = _submap_rotation_search(
        source, submap, pred,
        angle_range=angle_range,
        angle_step=angle_step,
        fine_step=sub_rot_fine,
        voxel_size=sub_rot_voxel,
    )

    return ICP(
        source, submap,
        error_threshold=icp_cfg.get("error_threshold", 1e-7),
        max_iterations=icp_cfg.get("max_iterations", 100),
        voxel_size=icp_cfg.get("voxel_size", 0.06),
        R_init=R_init, t_init=t_init,
        method="point_to_point",
        max_corr_dist=sub_corr_dist,
    )


# ── Loop-closure helpers ─────────────────────────────────────────────────────

def _find_loop_candidates(current_pose, scan_history, current_idx,
                          distance_threshold, min_interval, max_candidates,
                          min_cumulative_travel=10.0):
    """Return indices of historical scans that are spatially close but
    temporally far from the current scan — i.e. loop-closure candidates.

    An extra *min_cumulative_travel* gate ensures the vehicle has actually
    **left the area and come back** rather than just sitting still for
    ``min_interval`` scans (which would produce false loop closures that
    pin the trajectory to the origin).
    """
    curr_pos = current_pose[:2, 2]

    # Pre-compute cumulative odometric distance along the trajectory so we
    # can cheaply look up "how far did we drive between scan i and now?".
    # cum_dist[k] = cumulative path length from scan 0 to scan k.
    n = len(scan_history)
    cum_dist = np.zeros(n, dtype=np.float64)
    for k in range(1, n):
        cum_dist[k] = cum_dist[k - 1] + np.linalg.norm(
            scan_history[k][1][:2, 2] - scan_history[k - 1][1][:2, 2]
        )

    candidates = []
    for idx, (_, pose) in enumerate(scan_history):
        if current_idx - idx < min_interval:
            continue
        # Euclidean distance (current → candidate) must be small
        dist = np.linalg.norm(curr_pos - pose[:2, 2])
        if dist >= distance_threshold:
            continue
        # Cumulative travel between candidate and current scan must be
        # large — otherwise the robot never actually left this area.
        travel = cum_dist[current_idx] - cum_dist[idx] if current_idx < n else 0.0
        if travel < min_cumulative_travel:
            continue
        candidates.append((idx, dist))
    candidates.sort(key=lambda x: x[1])
    return candidates[:max_candidates]


def _rebuild_map(mapper, scan_history, scan_history_3d=None):
    """Clear the occupancy grid and replay every scan with its current pose."""
    mapper.reset()
    for k, (pts_2d, pose) in enumerate(scan_history):
        origin = pose[:2, 2]
        if scan_history_3d is not None:
            global_pts = transform_points_25d(scan_history_3d[k], pose)
        else:
            global_pts = transform_points_2d(pts_2d, pose)
        mapper.update_scan(origin, global_pts)


# ── main SLAM loop ───────────────────────────────────────────────────────────

def run_slam(cfg):
    icp_cfg  = cfg.get("icp", {})
    filt_cfg = cfg.get("filter", {})
    map_cfg  = cfg.get("mapping", {})
    svc_cfg  = cfg.get("service", {})
    disp_cfg = cfg.get("display", {})
    feat_cfg = cfg.get("features", {})
    lc_cfg   = cfg.get("loop_closure", {})
    sub_cfg  = cfg.get("submap", {})

    error_reject_threshold = icp_cfg.get("error_reject_threshold", 0.5)
    alignment_method = feat_cfg.get("method", "rotation_search")

    submap_enabled  = sub_cfg.get("enabled", True)
    submap_size     = sub_cfg.get("size", 30)
    submap_voxel    = sub_cfg.get("voxel_size", 0.06)
    sub_rot_range   = sub_cfg.get("rotation_range", 90.0)
    sub_rot_step    = sub_cfg.get("rotation_step", 1.0)
    sub_rot_fine    = sub_cfg.get("rotation_fine_step", 0.2)
    sub_rot_voxel   = sub_cfg.get("rotation_voxel_size", 0.25)
    sub_corr_dist   = sub_cfg.get("max_corr_dist", 0.5)

    z_min = filt_cfg.get("z_min", 0.2)
    z_max = filt_cfg.get("z_max", 2.0)

    map_resolution = map_cfg.get("resolution", 0.1)
    map_margin     = map_cfg.get("margin", 50.0)
    p_hit          = map_cfg.get("p_hit", 0.7)
    p_miss         = map_cfg.get("p_miss", 0.4)
    log_odds_min   = map_cfg.get("log_odds_min", -5.0)
    log_odds_max   = map_cfg.get("log_odds_max", 5.0)

    sleep_s = svc_cfg.get("sleep_s", 0.0)
    loop    = svc_cfg.get("loop", True)

    num_scans       = cfg.get("num_scans", None)
    process_every_n = cfg.get("process_every_n", 1)
    live_map   = disp_cfg.get("live_map", True)
    win_w      = disp_cfg.get("window_width", 1400)
    win_h      = disp_cfg.get("window_height", 1000)
    cmap       = disp_cfg.get("cmap", "gray")
    clim_min   = disp_cfg.get("clim_min", 0.0)
    clim_max   = disp_cfg.get("clim_max", 1.0)
    bg_color   = disp_cfg.get("background", "black")
    traj_color = disp_cfg.get("trajectory_color", "cyan")
    pose_color = disp_cfg.get("pose_color", "lime")
    pose_size  = disp_cfg.get("pose_size", 12)

    # Loop closure parameters
    lc_enabled       = lc_cfg.get("enabled", False)
    lc_distance      = lc_cfg.get("distance_threshold", 3.0)
    lc_min_interval  = lc_cfg.get("min_interval", 20)
    lc_max_cand      = lc_cfg.get("max_candidates", 3)
    lc_error_thresh  = lc_cfg.get("error_threshold", 0.03)
    lc_opt_iters     = lc_cfg.get("optimization_iterations", 20)
    lc_info_scale    = lc_cfg.get("information_scale", 10.0)
    lc_min_travel    = lc_cfg.get("min_cumulative_travel", 20.0)

    data_file = cfg.get("data_file", "data/ugvlidar-full.csv")

    # ── IMU configuration ─────────────────────────────────────────────
    imu_cfg     = cfg.get("imu", {})
    imu_enabled = imu_cfg.get("enabled", False)
    imu_file    = imu_cfg.get("file", "")
    imu_narrow  = imu_cfg.get("narrow_search_range", 5.0)   # degrees around IMU yaw
    imu = None
    imu_yaw_offset = 0.0          # calibrate IMU yaw to global frame
    if imu_enabled and imu_file:
        imu = IMUService(imu_file)

    # ── state ─────────────────────────────────────────────────────────
    service = LidarService(data_file, sleep_s=sleep_s, loop=loop)
    scan_stream = service.scans()
    global_pose = np.eye(3)
    pose_trajectory = []          # list of 3×3 matrices (for display)
    scan_history = []             # list of (points_2d, pose_3x3)
    prev_points = None
    mapper = None
    scans_processed = 0
    submap_buffer = []            # recent scans in global frame
    last_delta = None             # incremental pose for motion prediction
    prev_rel_time = None          # for IMU delta yaw

    scan_counter = 0              # raw scan counter (for process_every_n)

    # Pose graph
    pose_graph = PoseGraph2D()

    # Visualisation
    map_fig   = None
    map_grid  = None
    traj_mesh = None
    pose_mesh = None

    try:
        for timestamp, rel_time_us, raw_points in scan_stream:
            # ── scan skip (process every Nth) ─────────────────────────
            scan_counter += 1
            if process_every_n > 1 and (scan_counter % process_every_n) != 1:
                continue

            points = filter_and_flatten(raw_points, z_min=z_min, z_max=z_max)
            if points.shape[0] < 10:
                continue

            # ── first scan (initialisation) ───────────────────────────
            if prev_points is None:
                prev_points = points
                prev_rel_time = rel_time_us

                # Calibrate: record IMU yaw at the first scan so the
                # global frame starts at yaw = 0.
                if imu is not None:
                    imu_yaw_offset = imu.yaw_at(rel_time_us)
                    print(f"  [IMU] Calibrated initial yaw offset: "
                          f"{np.degrees(imu_yaw_offset):.1f}°")
                min_x, max_x, min_y, max_y = compute_bounds_from_scan(points, margin=map_margin)
                mapper = OccupancyGrid2D(
                    min_x=min_x, max_x=max_x,
                    min_y=min_y, max_y=max_y,
                    resolution=map_resolution,
                    p_hit=p_hit, p_miss=p_miss,
                    log_odds_min=log_odds_min, log_odds_max=log_odds_max,
                )
                sensor_origin = global_pose[:2, 2]
                global_points = transform_points_2d(points, global_pose)
                mapper.update_scan(sensor_origin, global_points)

                if submap_enabled:
                    submap_buffer.append(global_points.copy())

                scan_history.append((points.copy(), global_pose.copy()))
                pose_graph.add_node(pose_matrix_to_vec(global_pose))

                if live_map:
                    map_fig = pv.Plotter(window_size=(win_w, win_h))
                    map_grid = mapper.create_pyvista_grid()
                    map_fig.set_background(bg_color)
                    map_fig.add_mesh(
                        map_grid,
                        scalars="occ",
                        clim=(clim_min, clim_max),
                        cmap=cmap,
                        show_scalar_bar=False,
                    )
                    init_pt = np.array([[global_pose[0, 2], global_pose[1, 2], 0.0]])
                    traj_mesh = pv.PolyData(init_pt)
                    traj_mesh.lines = np.array([1, 0])
                    map_fig.add_mesh(traj_mesh, color=traj_color, line_width=2.0)
                    pose_mesh = pv.PolyData(init_pt.copy())
                    map_fig.add_mesh(
                        pose_mesh,
                        color=pose_color,
                        point_size=pose_size,
                        render_points_as_spheres=True,
                    )
                    map_fig.view_xy()
                    map_fig.reset_camera()
                    map_fig.enable_parallel_projection()

                    def zoom_in():
                        map_fig.camera.parallel_scale *= 0.9
                        map_fig.render()
                    def zoom_out():
                        map_fig.camera.parallel_scale *= 1.1
                        map_fig.render()
                    map_fig.add_key_event("plus", zoom_in)
                    map_fig.add_key_event("equal", zoom_in)
                    map_fig.add_key_event("minus", zoom_out)

                    map_fig.show(interactive_update=True, auto_close=False)
                continue

            # ── IMU yaw for this scan ─────────────────────────────────
            imu_yaw = None
            imu_delta = None
            if imu is not None:
                # Calibrated absolute yaw: 0 at first scan, relative after
                raw_yaw = imu.yaw_at(rel_time_us)
                imu_yaw = (raw_yaw - imu_yaw_offset + np.pi) % (2 * np.pi) - np.pi
                if prev_rel_time is not None:
                    imu_delta = imu.delta_yaw(prev_rel_time, rel_time_us)

            # ── Step 1: Scan-to-scan ICP (primary odometry) ───────────
            # This always runs first and gives the real incremental motion.
            if imu_delta is not None:
                ca, sa = np.cos(imu_delta), np.sin(imu_delta)
                R_imu = np.array([[ca, -sa], [sa, ca]])
                t_imu = np.zeros(2)
                r_inc, t_inc, err_inc = ICP(
                    prev_points, points,
                    error_threshold=icp_cfg.get("error_threshold", 1e-7),
                    max_iterations=icp_cfg.get("max_iterations", 100),
                    voxel_size=icp_cfg.get("voxel_size", 0.06),
                    R_init=R_imu, t_init=t_imu,
                    method=icp_cfg.get("method", "point_to_line"),
                    normal_k=icp_cfg.get("normal_k", 10),
                )
            else:
                r_inc, t_inc, err_inc = _run_icp_pair(
                    prev_points, points, icp_cfg, feat_cfg, alignment_method,
                )

            if err_inc > error_reject_threshold:
                print(f"Scan {scans_processed}: S2S error {err_inc:.6f} too high, skipping")
                prev_points = points
                prev_rel_time = rel_time_us
                scans_processed += 1
                continue

            # Accumulate incremental motion into global pose
            prev_global = global_pose.copy()
            global_pose = apply_incremental_pose_2d(global_pose, r_inc, t_inc)
            error = err_inc

            # ── Step 2: Submap drift correction (optional) ──────────
            # The submap ICP provides an absolute global pose.  We only
            # apply it if its answer is *close* to the incrementally
            # accumulated pose — preventing the submap from "resetting"
            # the pose back to the origin.
            if submap_enabled and len(submap_buffer) > 0:
                submap = _build_submap(submap_buffer, submap_voxel)

                r_sub, t_sub, err_sub = _attempt_submap_icp(
                    points, submap, global_pose.copy(),
                    imu_yaw, imu_narrow,
                    sub_rot_range, sub_rot_step, sub_rot_fine,
                    sub_rot_voxel, icp_cfg, sub_corr_dist,
                )

                if err_sub <= error_reject_threshold:
                    # Build submap-based absolute pose
                    submap_pose = np.eye(3)
                    submap_pose[:2, :2] = r_sub
                    submap_pose[:2, 2]  = t_sub

                    # Only apply the correction if the submap agrees
                    # roughly with the incrementally accumulated pose.
                    pos_diff = np.linalg.norm(
                        submap_pose[:2, 2] - global_pose[:2, 2])
                    sub_yaw = np.arctan2(r_sub[1, 0], r_sub[0, 0])
                    inc_yaw = np.arctan2(
                        global_pose[1, 0], global_pose[0, 0])
                    yaw_diff = abs(
                        (sub_yaw - inc_yaw + np.pi) % (2 * np.pi) - np.pi)

                    max_pos_corr = sub_corr_dist   # metres
                    max_yaw_corr = np.deg2rad(15.0)

                    if pos_diff < max_pos_corr and yaw_diff < max_yaw_corr:
                        global_pose = submap_pose
                        error = err_sub
                        print(f"  Submap correction applied "
                              f"(Δpos={pos_diff:.3f}m, "
                              f"Δyaw={np.degrees(yaw_diff):.1f}°)")

            last_delta = np.linalg.inv(prev_global) @ global_pose

            pose_trajectory.append(global_pose.copy())

            # ── pose graph: add node + odometry edge ──────────────────
            cur_idx = pose_graph.add_node(pose_matrix_to_vec(global_pose))
            prev_idx = cur_idx - 1
            z_odom = relative_transform_vec(
                scan_history[prev_idx][1], global_pose,
            )
            odom_info = np.eye(3) / max(error, 1e-6)
            pose_graph.add_edge(prev_idx, cur_idx, z_odom, odom_info)

            # ── update map incrementally ──────────────────────────────
            sensor_origin = global_pose[:2, 2]
            global_points = transform_points_2d(points, global_pose)
            scan_history.append((points.copy(), global_pose.copy()))

            if mapper is not None:
                mapper.update_scan(sensor_origin, global_points)

            if submap_enabled:
                submap_buffer.append(global_points.copy())
                if len(submap_buffer) > submap_size:
                    submap_buffer.pop(0)

            # ── loop closure ──────────────────────────────────────────
            lc_happened = False
            if lc_enabled and cur_idx >= lc_min_interval:
                candidates = _find_loop_candidates(
                    global_pose, scan_history, cur_idx,
                    lc_distance, lc_min_interval, lc_max_cand,
                    min_cumulative_travel=lc_min_travel,
                )
                if candidates:
                    print(f"  LC candidates for scan {cur_idx}: "
                          + ", ".join(f"#{ci}({cd:.1f}m)" for ci, cd in candidates))
                for (cand_idx, cand_dist) in candidates:
                    cand_points = scan_history[cand_idx][0]
                    r_lc, t_lc, err_lc = _run_icp_pair(
                        points, cand_points, icp_cfg, feat_cfg, alignment_method,
                    )
                    print(f"    LC scan {cur_idx}↔{cand_idx}: "
                          f"icp_err={err_lc:.6f}  {'✓' if err_lc < lc_error_thresh else '✗'}")
                    if err_lc < lc_error_thresh:
                        # ICP gives: R_lc @ cur + t_lc ≈ cand  (cur → cand)
                        # Pose graph edge (cur, cand) needs z = T_cur⁻¹ T_cand
                        # T_lc transforms cur → cand, so z = T_lc⁻¹
                        T_lc = np.eye(3)
                        T_lc[:2, :2] = r_lc
                        T_lc[:2, 2]  = t_lc
                        T_lc_inv = np.linalg.inv(T_lc)
                        z_lc = pose_matrix_to_vec(T_lc_inv)

                        lc_info = np.eye(3) * lc_info_scale / max(err_lc, 1e-6)
                        pose_graph.add_edge(cur_idx, cand_idx, z_lc, lc_info)
                        print(f"  ★ Loop closure accepted: scan {cur_idx} ↔ scan {cand_idx}  "
                              f"(dist={cand_dist:.2f}m, icp_err={err_lc:.6f})")
                        lc_happened = True
                        break   # one closure per scan is enough

                if lc_happened:
                    print("  Optimising pose graph …")
                    pose_graph.optimize(
                        n_iterations=lc_opt_iters, fix_node=0,
                    )
                    # Apply corrected poses back
                    corrected = pose_graph.get_poses_as_matrices()
                    for k in range(len(scan_history)):
                        scan_history[k] = (scan_history[k][0], corrected[k])
                    global_pose = corrected[-1]
                    pose_trajectory = [p for (_, p) in scan_history[1:]]

                    # Rebuild the submap buffer with corrected poses
                    if submap_enabled:
                        submap_buffer.clear()
                        for pts, pose in scan_history[-submap_size:]:
                            submap_buffer.append(transform_points_2d(pts, pose))

                    # Rebuild the occupancy grid from scratch
                    if mapper is not None:
                        print("  Rebuilding occupancy grid …")
                        _rebuild_map(mapper, scan_history)

            # ── live visualisation update ─────────────────────────────
            if live_map and map_grid is not None and mapper is not None:
                mapper.update_pyvista_grid(map_grid)
                map_fig.update_scalars(
                    map_grid.cell_data["occ"], mesh=map_grid, render=False,
                )
                positions = np.array([[p[0, 2], p[1, 2]] for p in pose_trajectory])
                if positions.size > 0:
                    positions_3d = np.column_stack(
                        (positions, np.zeros(len(positions)))
                    )
                    traj_mesh.points = positions_3d
                    traj_mesh.lines = np.hstack(
                        ([len(positions_3d)], np.arange(len(positions_3d)))
                    )
                    pose_mesh.points = positions_3d[-1:].copy()
                map_fig.render()
                map_fig.iren.process_events()

            prev_points = points
            prev_rel_time = rel_time_us
            scans_processed += 1
            pos = global_pose[:2, 2]
            yaw_deg = np.degrees(np.arctan2(global_pose[1, 0], global_pose[0, 0]))
            print(f"Scan {scans_processed:4d}  err={error:.6f}  "
                  f"pos=({pos[0]:+.3f}, {pos[1]:+.3f})  yaw={yaw_deg:+.2f}°")
            if num_scans is not None and scans_processed >= num_scans:
                break

    except KeyboardInterrupt:
        print("Stopping SLAM loop...")
    finally:
        if map_fig is not None:
            map_fig.close()

    return global_pose, pose_trajectory, mapper


def main():
    parser = argparse.ArgumentParser(description="Run 2D SLAM with ICP and occupancy mapping")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to YAML configuration file (default: config.yaml)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_cfg = cfg.get("output", {})

    global_pose, pose_trajectory, mapper = run_slam(cfg)

    print("global_pose:\n", global_pose)

    if mapper is not None:
        mapper.save_csv(out_cfg.get("csv", "tmp/occupancy_grid.csv"))
        mapper.save_npy(out_cfg.get("npy", "tmp/occupancy_grid.npy"))


if __name__ == "__main__":
    main()
