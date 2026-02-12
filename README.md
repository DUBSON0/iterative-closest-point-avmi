# 2D LiDAR SLAM with Iterative Closest Point

A from-scratch 2D SLAM (Simultaneous Localisation and Mapping) system for 3D LiDAR point clouds. Raw 3D scans are z-filtered and projected to 2D, then registered with ICP to build a real-time occupancy grid map with live trajectory visualisation.

## Features

- **Point-to-Point & Point-to-Line ICP** — classic SVD-based and linearised point-to-line solvers with voxel downsampling and KDTree-accelerated correspondence search
- **Pre-alignment for large rotations** — correlative rotation search (brute-force angle sweep) and/or feature-based alignment (curvature keypoints + sorted-distance descriptors + RANSAC) to escape local minima
- **Scan-to-submap matching** — aligns each scan against a rolling window of recent scans in the global frame to reduce accumulated drift
- **IMU fusion** — optional quaternion-based IMU orientation data constrains the rotation search, making aggressive turns robust
- **Loop closure & pose-graph optimisation** — spatial proximity detection with cumulative-travel gating, ICP verification, and Gauss-Newton optimisation on SE(2) to correct drift globally
- **Probabilistic occupancy grid** — log-odds Bresenham ray tracing with configurable hit/miss probabilities and 2.5D elevation tracking
- **Live visualisation** — real-time PyVista rendering of the occupancy map, robot trajectory, and current pose

## Project Structure

```
├── slam.py                  # Main SLAM loop (entry point)
├── config.yaml              # All tuneable parameters (ICP, mapping, display, …)
├── services/
│   ├── lidar_service.py     # Streams lidar scans from semicolon-delimited CSV
│   └── imu_service.py       # Loads IMU quaternions and provides yaw lookup
├── utilities/
│   ├── icp.py               # ICP solver, voxel downsampling, normal estimation
│   ├── features.py          # Curvature keypoints, descriptors, RANSAC, rotation search
│   ├── mapping.py           # OccupancyGrid2D with Bresenham ray tracing
│   └── pose_graph.py        # 2D pose-graph with Gauss-Newton optimiser
├── meta-utils/
│   ├── pcplayer.py          # Animated point cloud playback tool
│   ├── pcman.py             # Point cloud transform & export tool
│   └── pcview.py            # Static point cloud / trajectory viewer
└── data/                    # LiDAR & IMU CSV files (gitignored)
```

## Requirements

- Python 3.10+
- NumPy
- SciPy
- PyVista
- PyYAML

Install dependencies:

```bash
pip install numpy scipy pyvista pyyaml
```

## Usage

### Running SLAM

```bash
python slam.py                        # uses config.yaml
python slam.py --config my_config.yaml
```

The SLAM loop reads scans from the CSV specified in `config.yaml`, registers them with ICP, updates the occupancy grid, and (optionally) displays a live map. On exit the occupancy grid is saved to `tmp/`.

### Configuration

All parameters are controlled via `config.yaml`:

| Section | Key parameters |
|---|---|
| **icp** | `method` (point_to_point / point_to_line), `voxel_size`, `max_iterations`, `error_reject_threshold` |
| **features** | `method` (rotation_search / features / both / none), angle steps, RANSAC settings |
| **submap** | `enabled`, `size` (window length), rotation search range |
| **loop_closure** | `enabled`, `distance_threshold`, `min_interval`, `min_cumulative_travel` |
| **imu** | `enabled`, `file`, `narrow_search_range` |
| **mapping** | `resolution`, `p_hit`, `p_miss`, log-odds clamp range |
| **filter** | `z_min` / `z_max` — height slice before 2D projection |
| **display** | `live_map`, window size, colours |

### Data Format

**LiDAR CSV** — one scan per line, semicolon-delimited:

```
timestamp;x1;y1;z1;x2;y2;z2;...
```

**IMU CSV** — one reading per line, semicolon-delimited:

```
timestamp_us;qx;qy;qz;qw
```

### Meta-Utilities

```bash
# Animated lidar playback
python meta-utils/pcplayer.py                       # interactive file picker
python meta-utils/pcplayer.py data/file.csv --fps 30

# Transform & export a point cloud
python meta-utils/pcman.py data/teapot.csv --rotate-z 45 --scale 2.0

# Static point cloud viewer
python meta-utils/pcview.py data/teapot.csv --color cyan
```

## Pipeline Overview

```
Raw 3D scan
    │
    ▼
Z-filter & flatten to 2D
    │
    ▼
Pre-alignment (rotation search / features)
    │
    ▼
Scan-to-scan ICP  ──►  Incremental pose
    │
    ▼
Scan-to-submap ICP (drift correction)
    │
    ▼
Pose-graph update  ◄──  Loop closure detection
    │
    ▼
Occupancy grid update (Bresenham ray tracing)
    │
    ▼
Live map render (PyVista)
```
