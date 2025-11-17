import argparse
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Teapot point cloud viewer")
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
    parser.add_argument("--export", default=str(export_path))
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


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    centers = (mins + maxs) / 2.0
    span = max((maxs - mins).max() / 2.0, 1e-6)
    ax.set_xlim(centers[0] - span, centers[0] + span)
    ax.set_ylim(centers[1] - span, centers[1] + span)
    ax.set_zlim(centers[2] - span, centers[2] + span)


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
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    scatter = ax.scatter(
        transformed[:, 0],
        transformed[:, 1],
        transformed[:, 2],
        c=transformed[:, 2],
        s=args.point_size,
        cmap=args.cmap,
        marker=".",
    )
    fig.colorbar(scatter, ax=ax, label="z")
    ax.set_title(f"Transformed Teapot ({len(transformed)} points)")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_equal_axes(ax, points)
    plt.show()


if __name__ == "__main__":
    main()

