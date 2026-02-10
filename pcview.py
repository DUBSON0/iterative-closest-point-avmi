#!/usr/bin/env python3
"""
Point Cloud Viewer using PyVista

Assumes each line in the file is one point: x,y,z.
"""

import argparse
import numpy as np
import pyvista as pv


def load_point_cloud(file_path, delimiter=','):
    points = np.loadtxt(file_path, delimiter=delimiter)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    if points.shape[1] != 3:
        raise ValueError(f"Expected 3 columns (x, y, z), got {points.shape[1]}")
    return points


def visualize_point_cloud(
    points,
    point_size=2.0,
    color='cyan',
    background_color='black',
    window_size=(800, 600),
):
    point_cloud = pv.PolyData(points)
    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color
    plotter.add_mesh(point_cloud, color=color, point_size=point_size)
    plotter.show()


def visualize_point_clouds(
    point_sets,
    labels=None,
    colors=None,
    point_size=2.0,
    background_color='black',
    window_size=(800, 600),
    show_legend=True,
    enable_toggles=False,
):
    if labels is None:
        labels = [f"Cloud {i + 1}" for i in range(len(point_sets))]
    if colors is None:
        colors = ['cyan'] * len(point_sets)
    if len(point_sets) != len(labels) or len(point_sets) != len(colors):
        raise ValueError("point_sets, labels, and colors must have the same length")

    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color
    actors = []
    for points, label, color in zip(point_sets, labels, colors):
        cloud = pv.PolyData(points)
        actor = plotter.add_mesh(cloud, color=color, point_size=point_size, label=label)
        actors.append(actor)

    if show_legend:
        plotter.add_legend()

    if enable_toggles:
        def toggle_actor(actor):
            def _toggle(flag):
                actor.SetVisibility(flag)
            return _toggle

        base_x = 10
        base_y = 10
        step_y = 25
        for idx, (actor, label) in enumerate(zip(actors, labels)):
            y = base_y + (idx * step_y)
            plotter.add_checkbox_button_widget(
                toggle_actor(actor),
                value=True,
                position=(base_x, y),
            )
            plotter.add_text(label, position=(base_x + 30, y - 5), font_size=10)

    plotter.show()


def visualize_trajectory(
    poses,
    line_color='yellow',
    point_color='white',
    point_size=6.0,
    line_width=2.0,
    background_color='black',
    window_size=(800, 600),
):
    if not poses:
        raise ValueError("poses is empty")

    positions = np.array([pose[:3, 3] for pose in poses])
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("poses must be a list of 4x4 transforms")

    plotter = pv.Plotter(window_size=window_size)
    plotter.background_color = background_color

    trajectory_points = pv.PolyData(positions)
    plotter.add_mesh(trajectory_points, color=point_color, point_size=point_size)

    if len(positions) > 1:
        polyline = pv.lines_from_points(positions)
        plotter.add_mesh(polyline, color=line_color, line_width=line_width)

    plotter.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize point cloud data from CSV files using PyVista",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python point_cloud_viewer.py data/teapot.csv
  python point_cloud_viewer.py data/teapot.csv --point-size 2 --color cyan
        """
    )
    
    parser.add_argument(
        'file_path',
        type=str,
        help='Path to the CSV file containing x,y,z point cloud data'
    )
    
    parser.add_argument(
        '--delimiter',
        type=str,
        default=',',
        help='Delimiter used in the CSV file (default: ",")'
    )

    parser.add_argument(
        '--point-size',
        type=float,
        default=2.0,
        help='Size of points in the visualization (default: 2.0)'
    )
    
    parser.add_argument(
        '--color',
        type=str,
        default='cyan',
        help='Color of the points (default: "cyan"). Can be any valid color name or hex code.'
    )

    parser.add_argument(
        '--background',
        type=str,
        default='black',
        help='Background color (default: "black")'
    )
    
    parser.add_argument(
        '--window-size',
        type=int,
        nargs=2,
        default=[800, 600],
        metavar=('WIDTH', 'HEIGHT'),
        help='Window size in pixels (default: 800 600)'
    )
    
    args = parser.parse_args()
    
    # Load point cloud
    print(f"Loading point cloud from: {args.file_path}")
    points = load_point_cloud(
        args.file_path,
        delimiter=args.delimiter,
    )
    print(f"Loaded {len(points)} points")
    print(f"Point cloud bounds: X=[{points[:, 0].min():.3f}, {points[:, 0].max():.3f}], "
          f"Y=[{points[:, 1].min():.3f}, {points[:, 1].max():.3f}], "
          f"Z=[{points[:, 2].min():.3f}, {points[:, 2].max():.3f}]")
    
    # Visualize
    visualize_point_cloud(
        points,
        point_size=args.point_size,
        color=args.color,
        background_color=args.background,
        window_size=tuple(args.window_size),
    )


if __name__ == '__main__':
    main()
