import argparse
import yaml
import numpy as np
import pyvista as pv

import lidar_service
import mapping
from icp import ICP
from features import feature_based_alignment, rotation_search


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def filter_and_flatten(points, z_min=0.2, z_max=2.0):
    """Keep only points with z in [z_min, z_max], then return x,y only."""
    mask = (points[:, 2] >= z_min) & (points[:, 2] <= z_max)
    return points[mask, :2].copy()


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


def run_slam(cfg):
    icp_cfg = cfg.get("icp", {})
    filt_cfg = cfg.get("filter", {})
    map_cfg = cfg.get("mapping", {})
    svc_cfg = cfg.get("service", {})
    disp_cfg = cfg.get("display", {})

    error_threshold = icp_cfg.get("error_threshold", 1e-7)
    max_iterations = icp_cfg.get("max_iterations", 5000)
    voxel_size = icp_cfg.get("voxel_size", 0.01)
    error_reject_threshold = icp_cfg.get("error_reject_threshold", 0.5)

    feat_cfg = cfg.get("features", {})
    # alignment_method: "rotation_search" | "features" | "both" | "none"
    alignment_method = feat_cfg.get("method", "rotation_search")

    z_min = filt_cfg.get("z_min", 0.2)
    z_max = filt_cfg.get("z_max", 2.0)

    map_resolution = map_cfg.get("resolution", 0.1)
    map_margin = map_cfg.get("margin", 50.0)
    p_hit = map_cfg.get("p_hit", 0.7)
    p_miss = map_cfg.get("p_miss", 0.4)
    log_odds_min = map_cfg.get("log_odds_min", -5.0)
    log_odds_max = map_cfg.get("log_odds_max", 5.0)

    sleep_s = svc_cfg.get("sleep_s", 0.0)
    loop = svc_cfg.get("loop", True)

    num_scans = cfg.get("num_scans", None)
    live_map = disp_cfg.get("live_map", True)
    win_w = disp_cfg.get("window_width", 1400)
    win_h = disp_cfg.get("window_height", 1000)
    cmap = disp_cfg.get("cmap", "gray")
    clim_min = disp_cfg.get("clim_min", 0.0)
    clim_max = disp_cfg.get("clim_max", 1.0)
    bg_color = disp_cfg.get("background", "black")
    traj_color = disp_cfg.get("trajectory_color", "cyan")
    pose_color = disp_cfg.get("pose_color", "lime")
    pose_size = disp_cfg.get("pose_size", 12)

    data_file = cfg.get("data_file", "data/ugvlidar-full.csv")

    service = lidar_service.LidarService(data_file, sleep_s=sleep_s, loop=loop)
    scan_stream = service.scans()
    global_pose = np.eye(3)
    pose_trajectory = []
    prev_points = None
    mapper = None
    scans_processed = 0
    map_fig = None
    map_grid = None
    traj_mesh = None
    pose_mesh = None

    try:
        for timestamp, raw_points in scan_stream:
            points = filter_and_flatten(raw_points, z_min=z_min, z_max=z_max)
            if points.shape[0] < 10:
                continue

            if prev_points is None:
                prev_points = points
                min_x, max_x, min_y, max_y = compute_bounds_from_scan(points, margin=map_margin)
                mapper = mapping.OccupancyGrid2D(
                    min_x=min_x,
                    max_x=max_x,
                    min_y=min_y,
                    max_y=max_y,
                    resolution=map_resolution,
                    p_hit=p_hit,
                    p_miss=p_miss,
                    log_odds_min=log_odds_min,
                    log_odds_max=log_odds_max,
                )
                sensor_origin = global_pose[:2, 2]
                global_points = transform_points_2d(points, global_pose)
                mapper.update_scan(sensor_origin, global_points)
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
                    init_point = np.array([[global_pose[0, 2], global_pose[1, 2], 0.0]])
                    traj_mesh = pv.PolyData(init_point)
                    traj_mesh.lines = np.array([1, 0])
                    map_fig.add_mesh(traj_mesh, color=traj_color, line_width=2.0)
                    pose_mesh = pv.PolyData(init_point.copy())
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

            # ── coarse pre-alignment (initial guess for ICP) ─────────────
            R_init, t_init = None, None

            if alignment_method in ("rotation_search", "both"):
                R_init, t_init, _ = rotation_search(
                    prev_points,
                    points,
                    voxel_size=feat_cfg.get("rotation_voxel_size", 0.3),
                    angle_step_coarse=feat_cfg.get("angle_step_coarse", 2.0),
                    angle_step_fine=feat_cfg.get("angle_step_fine", 0.2),
                )

            if alignment_method in ("features", "both"):
                # If "both", feature alignment refines the rotation-search
                # result, so we pre-transform source first.
                fa_src = prev_points
                if R_init is not None:
                    fa_src = prev_points @ R_init.T + t_init

                R_feat, t_feat, n_inliers = feature_based_alignment(
                    fa_src,
                    points,
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
                        # Compose: feature ∘ rotation_search
                        R_init = R_feat @ R_init
                        t_init = t_init @ R_feat.T + t_feat
                    else:
                        R_init, t_init = R_feat, t_feat
                # else: keep rotation_search result (or None)

            # ── ICP refinement ───────────────────────────────────────────
            r, t, error = ICP(
                prev_points,
                points,
                error_threshold=error_threshold,
                max_iterations=max_iterations,
                voxel_size=voxel_size,
                R_init=R_init,
                t_init=t_init,
            )
            if error > error_reject_threshold:
                print(f"Scan {scans_processed}: error {error:.6f} too high, skipping")
                prev_points = points
                scans_processed += 1
                continue
            global_pose = apply_incremental_pose_2d(global_pose, r, t)
            pose_trajectory.append(global_pose.copy())

            sensor_origin = global_pose[:2, 2]
            global_points = transform_points_2d(points, global_pose)
            if mapper is not None:
                mapper.update_scan(sensor_origin, global_points)
                if live_map and map_grid is not None:
                    mapper.update_pyvista_grid(map_grid)
                    map_fig.update_scalars(map_grid.cell_data["occ"], mesh=map_grid, render=False)
                    positions = np.array([[p[0, 2], p[1, 2]] for p in pose_trajectory])
                    if positions.size > 0:
                        positions_3d = np.column_stack((positions, np.zeros(len(positions))))
                        traj_mesh.points = positions_3d
                        traj_mesh.lines = np.hstack(([len(positions_3d)], np.arange(len(positions_3d))))
                        pose_mesh.points = positions_3d[-1:].copy()
                    map_fig.render()
                    map_fig.iren.process_events()

            prev_points = points
            scans_processed += 1
            if num_scans is not None and scans_processed >= num_scans:
                break
            print("Scan: ", scans_processed, "Error: ", error)
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
        mapper.save_csv(out_cfg.get("csv", "occupancy_grid.csv"))
        mapper.save_npy(out_cfg.get("npy", "occupancy_grid.npy"))


if __name__ == "__main__":
    main()
