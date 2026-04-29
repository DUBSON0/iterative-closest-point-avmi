#!/usr/bin/env python3
"""
Teapot ICP Demo
===============
A single window split into two adjacent panels:

  Left  — Original (blue) and transformed (red) before alignment.
  Right — Original (blue) and ICP-aligned (green) after alignment.

Run from the repo root:
    python demos/teapot_icp_demo.py
"""

import sys
import os
import numpy as np
import pyvista as pv
from scipy.spatial import KDTree

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utilities.icp import ICP


# ── 1. Load ──────────────────────────────────────────────────────────────────

teapot_path = os.path.join(ROOT, "teapot.csv")
original = np.loadtxt(teapot_path, delimiter=",")
print(f"Loaded {len(original)} points from teapot.csv")
print(f"  Bounds  X=[{original[:,0].min():.3f}, {original[:,0].max():.3f}]"
      f"  Y=[{original[:,1].min():.3f}, {original[:,1].max():.3f}]"
      f"  Z=[{original[:,2].min():.3f}, {original[:,2].max():.3f}]")


# ── 2. Apply transformation ───────────────────────────────────────────────────

angle_deg = 25.0
angle = np.radians(angle_deg)
Ry = np.array([
    [ np.cos(angle), 0, np.sin(angle)],
    [             0, 1,             0],
    [-np.sin(angle), 0, np.cos(angle)],
])
translation = np.array([0.25, 0.05, 0.0])

transformed = original @ Ry.T + translation

print(f"\nApplied transformation:")
print(f"  Rotation    : {angle_deg}° around the Y-axis")
print(f"  Translation : {translation}")


# ── 3. Run ICP ───────────────────────────────────────────────────────────────

print("\n[ICP]  Aligning transformed → original …")

R_icp, t_icp, _ = ICP(
    source=transformed,
    target=original,
    error_threshold=1e-12,
    max_iterations=300,
    voxel_size=0.005,
    method="point_to_point",
)

aligned = transformed @ R_icp.T + t_icp

tree = KDTree(original)
dists, _ = tree.query(aligned)
print(f"  Mean residual : {dists.mean():.6f}")
print(f"  Max  residual : {dists.max():.6f}")


# ── 4. Two-panel window ───────────────────────────────────────────────────────

POINT_SIZE  = 14
BG          = "white"
TITLE_COLOR = "black"

def add_cloud(plotter, points, color, label):
    cloud = pv.PolyData(points)
    plotter.add_mesh(
        cloud, color=color, point_size=POINT_SIZE,
        render_points_as_spheres=True, label=label,
    )

plotter = pv.Plotter(shape=(1, 2), window_size=(1400, 700))
plotter.background_color = BG

LEGEND_SIZE = (0.26, 0.11)
TITLE_FONT  = 14

# ── Left panel: before ───────────────────────────────────────────────────────
plotter.subplot(0, 0)
plotter.background_color = BG
add_cloud(plotter, original,    "dodgerblue", "Original")
add_cloud(plotter, transformed, "tomato",     "Transformed")
plotter.add_legend(size=LEGEND_SIZE, face="circle", loc="lower left")
plotter.add_text("Before ICP", position="upper_edge", font_size=TITLE_FONT, color=TITLE_COLOR)
plotter.reset_camera()

# ── Right panel: after ───────────────────────────────────────────────────────
plotter.subplot(0, 1)
plotter.background_color = BG
add_cloud(plotter, original, "dodgerblue", "Original")
add_cloud(plotter, aligned,  "limegreen",  "Transformed")
plotter.add_legend(size=LEGEND_SIZE, face="circle", loc="lower left")
plotter.add_text("After ICP",  position="upper_edge", font_size=TITLE_FONT, color=TITLE_COLOR)
plotter.reset_camera()

print("\n[Window]  Left: before alignment  |  Right: after alignment")
plotter.show()
