import argparse
from pathlib import Path

import numpy as np
import pyvista as pv


def parse_args():
    parser = argparse.ArgumentParser(description="PCMan point cloud viewer")
    base = Path(__file__).resolve().parent
    default_path = base / "data" / "teapot.csv"
    export_path = base / "data" / "tranformed-teapot.csv"
    parser.add_argument("cloud", nargs="?", default=str(default_path))
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--point-size", type=float, default=2.0)
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--rotate-x", type=float, default=0.0)
    parser.add_argument("--rotate-y", type=float, default=0.0)
    parser.add_argument("--rotate-z", type=float, default=0.0)
    parser.add_argument("--translate", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    parser.add_argument("--export", "--output", dest="export", default=str(export_path))
    return parser.parse_args()


def load_points(path, stride):
    if stride <= 0:
        raise ValueError("stride must be positive")
    data = np.loadtxt(path, delimiter=",")
    return data[::stride]


def rotation_matrix(rx, ry, rz):
    cx, cy, cz = np.cos(np.radians([rx, ry, rz]))
    sx, sy, sz = np.sin(np.radians([rx, ry, rz]))
    rx_mat = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    ry_mat = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rz_mat = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return rz_mat @ ry_mat @ rx_mat


def transform_points(points, scale, rotations, translation):
    scaled = points * scale
    rot = rotation_matrix(*rotations)
    rotated = scaled @ rot.T
    translated = rotated + np.asarray(translation)
    return translated


def export_points(points, path):
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, points, delimiter=",")


def main():
    args = parse_args()
    cloud_path = Path(args.cloud).expanduser()
    points = load_points(cloud_path, args.stride)
    transformed = transform_points(
        points,
        args.scale,
        (args.rotate_x, args.rotate_y, args.rotate_z),
        args.translate,
    )
    export_points(transformed, args.export)
    original_cloud = pv.PolyData(points)
    transformed_cloud = pv.PolyData(transformed)
    transformed_cloud["z"] = transformed[:, 2]
    plotter = pv.Plotter()
    plotter.add_mesh(
        original_cloud,
        color="black",
        point_size=args.point_size,
        render_points_as_spheres=False,
        label="Original",
    )
    plotter.add_mesh(
        transformed_cloud,
        color="red",
        point_size=args.point_size,
        render_points_as_spheres=False,
        label="Transformed",
    )
    plotter.add_axes(line_width=2, labels_off=False)
    plotter.add_legend()
    plotter.show_grid()
    plotter.add_text(
        f"Transformed Point Cloud ({len(transformed)} points)",
        font_size=12,
        position="upper_left",
    )
    plotter.show()


if __name__ == "__main__":
    main()

