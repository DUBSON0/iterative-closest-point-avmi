# Iterative Closest Point AVMI

2D SLAM pipeline built around ICP scan matching, optional IMU yaw fusion, submap
alignment, loop closure, and occupancy-grid mapping.

## Features

- ICP registration with both `point_to_line` and `point_to_point` modes
- Pre-alignment options: rotation search, feature matching, both, or none
- Optional IMU yaw integration to stabilize turns
- Submap-based scan matching to reduce drift
- Pose graph loop closure and optimization
- Live occupancy-grid visualization with trajectory display

## Project Structure

- `slam.py`: main SLAM runner and visualization loop
- `config.yaml`: all runtime parameters
- `utilities/`: ICP, features, mapping, pose-graph helpers
- `services/`: lidar and IMU data loaders
- `demos/teapot_icp_demo.py`: standalone ICP point-cloud alignment demo
- `tmp/`: generated map outputs (`.csv`, `.npy`)

## Requirements

- Python 3.10+ (recommended)
- `numpy`
- `scipy`
- `pyvista`
- `pyyaml`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running SLAM

From the repository root:

```bash
python slam.py --config config.yaml
```

At the end of the run, the occupancy grid is written to paths configured under
`output` in `config.yaml` (default: `tmp/occupancy_grid.csv` and
`tmp/occupancy_grid.npy`).

## Running the ICP Demo

The demo aligns a transformed teapot point cloud back to the original and opens
a split view (before/after ICP):

```bash
python demos/teapot_icp_demo.py
```

## Input Data Format

### Lidar file

Configured by `data_file` in `config.yaml`.

Each line:

```text
timestamp_us;x1;y1;z1;x2;y2;z2;...;xn;yn;zn
```

Semicolon-delimited values are parsed into XYZ triples per scan.

### IMU file (optional)

Enabled via `imu.enabled` and `imu.file` in `config.yaml`.

Each line:

```text
timestamp_us;qx;qy;qz;qw
```

Quaternions are converted to yaw and time-aligned with lidar scans.

## Key Configuration Sections

- `imu`: IMU integration and narrow yaw search band
- `icp`: solver method, convergence limits, voxel size, and rejection threshold
- `features`: pre-alignment strategy and feature/RANSAC parameters
- `submap`: rolling local map alignment settings
- `loop_closure`: candidate search, acceptance, and optimization behavior
- `filter`: z-range filtering before flattening to 2D
- `mapping`: occupancy-grid resolution and log-odds update parameters
- `display`: live map window and rendering options
- `output`: output map file paths

## Notes

- Default config assumes dataset files exist at `data/1007lidar.csv` and
  `data/1007imu.csv`.
- If you only want scan-to-scan ICP, set `submap.enabled: false`.
- For headless runs, set `display.live_map: false`.
