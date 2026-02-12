#!/usr/bin/env python3
"""
Point Cloud Player

Select a .csv from the data/ directory and play it back as an animated
point cloud using PyVista.  Supports two CSV flavours:

  1. Simple  – comma-delimited, one point per row (x,y,z).
     Displayed as a single static cloud.
  2. Lidar   – semicolon-delimited, one scan per row
     (timestamp;x1;y1;z1;x2;y2;z2;...).
     Played back frame-by-frame.

Down-sampling is applied via a configurable stride / voxel size so that
even very dense clouds render smoothly.
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

# ── Data directory ──────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ── Helpers ─────────────────────────────────────────────────────────────────
def _list_csvs(data_dir: Path) -> list[Path]:
    """Return sorted list of .csv files in *data_dir*."""
    return sorted(p for p in data_dir.iterdir() if p.suffix.lower() == ".csv")


def _pick_file(csvs: list[Path]) -> Path:
    """Interactive menu – let the user choose a file."""
    print("\nAvailable .csv files in data/:\n")
    for idx, p in enumerate(csvs, 1):
        size_mb = p.stat().st_size / 1e6
        print(f"  [{idx:>2}]  {p.name}  ({size_mb:.1f} MB)")
    print()
    while True:
        try:
            choice = int(input("Select a file number: "))
            if 1 <= choice <= len(csvs):
                return csvs[choice - 1]
        except (ValueError, EOFError):
            pass
        print(f"Please enter a number between 1 and {len(csvs)}.")


def _detect_format(path: Path) -> str:
    """Return 'lidar' or 'simple' based on the first line of the file."""
    with open(path) as f:
        first_line = f.readline()
    if ";" in first_line:
        return "lidar"
    return "simple"


# ── Loaders ─────────────────────────────────────────────────────────────────
def _parse_lidar_line(line: str) -> tuple[int, np.ndarray]:
    """Parse a semicolon-delimited lidar row → (timestamp, Nx3 points)."""
    parts = line.strip().split(";")
    timestamp = int(parts[0])
    floats = np.array([float(v) for v in parts[1:]], dtype=np.float64)
    if floats.size % 3 != 0:
        raise ValueError("Lidar row values are not a multiple of 3")
    points = floats.reshape(-1, 3)
    # Drop all-zero points (padding)
    mask = np.any(points != 0, axis=1)
    return timestamp, points[mask]


def load_lidar_frames(path: Path) -> list[tuple[int, np.ndarray]]:
    """Load all lidar frames from a semicolon-delimited CSV."""
    frames = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            frames.append(_parse_lidar_line(line))
    return frames


def load_simple_cloud(path: Path) -> np.ndarray:
    """Load a simple x,y,z-per-row CSV (comma-delimited)."""
    points = np.loadtxt(path, delimiter=",")
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] < 3:
        raise ValueError(f"Expected ≥3 columns, got {points.shape[1]}")
    return points[:, :3]


# ── Down-sampling ───────────────────────────────────────────────────────────
def downsample_stride(points: np.ndarray, stride: int) -> np.ndarray:
    """Thin a cloud by keeping every *stride*-th point."""
    if stride <= 1:
        return points
    return points[::stride]


def downsample_voxel(points: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel grid down-sampling – keep one point per voxel cell."""
    if voxel_size <= 0:
        return points
    mins = points.min(axis=0)
    keys = ((points - mins) / voxel_size).astype(np.int64)
    # Unique voxel keys → one representative point each
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def downsample(points: np.ndarray, stride: int = 1,
               voxel_size: float = 0.0) -> np.ndarray:
    """Apply stride then (optionally) voxel down-sampling."""
    pts = downsample_stride(points, stride)
    if voxel_size > 0:
        pts = downsample_voxel(pts, voxel_size)
    return pts


# ── Playback ────────────────────────────────────────────────────────────────
def play_lidar(frames: list[tuple[int, np.ndarray]],
               stride: int, voxel_size: float,
               point_size: float, fps: float,
               color: str, bg: str, window_size: tuple[int, int]):
    """Animate lidar frames in a PyVista window."""
    if not frames:
        print("No frames to play.")
        return

    delay = 1.0 / fps if fps > 0 else 0.0
    first_pts = downsample(frames[0][1], stride, voxel_size)
    cloud = pv.PolyData(first_pts)

    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = bg
    plotter.add_mesh(cloud, color=color, point_size=point_size,
                     render_points_as_spheres=False, name="cloud")
    plotter.add_axes(line_width=2, labels_off=False)

    title_actor = plotter.add_text(
        f"Scan 1/{len(frames)}  |  {len(first_pts)} pts",
        position="upper_left", font_size=12, color="white", name="title",
    )

    plotter.show(interactive_update=True, auto_close=False)

    for i, (ts, pts) in enumerate(frames):
        ds = downsample(pts, stride, voxel_size)
        # Rebuild mesh each frame (point count can change between scans)
        plotter.remove_actor("cloud")
        cloud = pv.PolyData(ds)
        plotter.add_mesh(cloud, color=color, point_size=point_size,
                         render_points_as_spheres=False, name="cloud")
        plotter.remove_actor("title")
        plotter.add_text(
            f"Scan {i + 1}/{len(frames)}  |  {len(ds)} pts  |  ts {ts}",
            position="upper_left", font_size=12, color="white", name="title",
        )
        plotter.update()
        if delay > 0:
            time.sleep(delay)

    # Keep window open after playback
    plotter.show(interactive_update=False, auto_close=False)


def show_static(points: np.ndarray,
                stride: int, voxel_size: float,
                point_size: float,
                color: str, bg: str, window_size: tuple[int, int]):
    """Display a static point cloud."""
    ds = downsample(points, stride, voxel_size)
    cloud = pv.PolyData(ds)
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = bg
    plotter.add_mesh(cloud, color=color, point_size=point_size,
                     render_points_as_spheres=False)
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_text(
        f"{len(ds)} points (downsampled from {len(points)})",
        position="upper_left", font_size=12, color="white",
    )
    plotter.show()


# ── CLI ─────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Point Cloud Player – select & play .csv files from data/",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("file", nargs="?", default=None,
                        help="Path to a .csv file (omit for interactive picker)")
    parser.add_argument("--data-dir", type=str, default=str(DATA_DIR),
                        help="Directory containing .csv files")
    parser.add_argument("--stride", type=int, default=1,
                        help="Keep every N-th point (default: 1 = keep all)")
    parser.add_argument("--voxel", type=float, default=0.0,
                        help="Voxel grid size for down-sampling (0 = off)")
    parser.add_argument("--point-size", type=float, default=2.0,
                        help="Rendered point size (default: 2.0)")
    parser.add_argument("--fps", type=float, default=30.0,
                        help="Target playback FPS for lidar files (default: 30)")
    parser.add_argument("--color", type=str, default="cyan",
                        help="Point colour (default: cyan)")
    parser.add_argument("--bg", type=str, default="black",
                        help="Background colour (default: black)")
    parser.add_argument("--window-size", type=int, nargs=2, default=[1280, 720],
                        metavar=("W", "H"),
                        help="Window size in pixels (default: 1280 720)")
    return parser.parse_args()


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # ── File selection ──────────────────────────────────────────────────
    if args.file:
        chosen = Path(args.file)
    else:
        csvs = _list_csvs(data_dir)
        if not csvs:
            print(f"No .csv files found in {data_dir}")
            sys.exit(1)
        chosen = _pick_file(csvs)

    if not chosen.exists():
        print(f"File not found: {chosen}")
        sys.exit(1)

    fmt = _detect_format(chosen)
    print(f"\nFile   : {chosen.name}")
    print(f"Format : {fmt}")

    ws = tuple(args.window_size)

    # ── Load & play ─────────────────────────────────────────────────────
    if fmt == "lidar":
        print("Loading lidar frames …")
        frames = load_lidar_frames(chosen)
        print(f"Loaded {len(frames)} frames")
        play_lidar(frames, args.stride, args.voxel,
                   args.point_size, args.fps,
                   args.color, args.bg, ws)
    else:
        print("Loading static point cloud …")
        points = load_simple_cloud(chosen)
        print(f"Loaded {len(points)} points")
        show_static(points, args.stride, args.voxel,
                    args.point_size,
                    args.color, args.bg, ws)


if __name__ == "__main__":
    main()
