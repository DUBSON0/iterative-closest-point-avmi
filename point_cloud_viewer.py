import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

def load_point_cloud(filename):
    """
    Load point cloud data from file where each point is represented by three consecutive lines:
    line 1: x coordinate
    line 2: y coordinate
    line 3: z coordinate
    """
    points = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    # Process lines in groups of 3
    for i in range(0, len(lines) - len(lines) % 3, 3):
        try:
            x = float(lines[i].strip())
            y = float(lines[i+1].strip())
            z = float(lines[i+2].strip())
            points.append([x, y, z])
        except (ValueError, IndexError):
            # Skip invalid data points
            continue

    return np.array(points)

def filter_outliers(points, max_range=60.0):
    """
    Filter out points where any coordinate exceeds the maximum lidar range.

    Args:
        points: numpy array of shape (n, 3) containing xyz coordinates
        max_range: maximum allowed value for any coordinate (default: 60.0)

    Returns:
        filtered numpy array with outliers removed
    """
    # Find points where all coordinates are within the valid range
    valid_mask = np.all(np.abs(points) <= max_range, axis=1)
    filtered_points = points[valid_mask]

    print(f"Removed {len(points) - len(filtered_points)} outlier points")
    print(f"Kept {len(filtered_points)} valid points")

    return filtered_points

def save_filtered_data(points, output_filename="filtered_point_cloud.txt"):
    """
    Save filtered point cloud data to a text file in the same format as input.
    Each point is saved as three consecutive lines (x, y, z).

    Args:
        points: numpy array of shape (n, 3)
        output_filename: name of output file
    """
    with open(output_filename, 'w') as f:
        for point in points:
            f.write(f"{point[0]}\n{point[1]}\n{point[2]}\n")

    print(f"Filtered data saved to {output_filename}")

def uniform_sample_points(points, num_samples=1000):
    """
    Uniformly sample points from the dataset to reduce visualization lag.

    Args:
        points: numpy array of shape (n, 3)
        num_samples: number of points to sample (default: 1000)

    Returns:
        numpy array with uniformly sampled points
    """
    if len(points) <= num_samples:
        return points

    # Generate uniform indices across the dataset
    indices = np.linspace(0, len(points) - 1, num_samples, dtype=int)
    sampled_points = points[indices]

    print(f"Uniformly sampled {len(sampled_points)} points from {len(points)} total points")
    return sampled_points

def plot_point_cloud(points):
    """
    Create a 3D scatter plot of the point cloud
    """
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', s=1, alpha=0.7)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
    cbar.set_label('Z Coordinate')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Filtered Point Cloud Viewer\n{len(points)} sampled points (from filtered dataset)')

    # Set equal aspect ratio
    max_range = np.array([x.max()-x.min(), y.max()-y.min(), z.max()-z.min()]).max() / 2.0
    mid_x = (x.max()+x.min()) * 0.5
    mid_y = (y.max()+y.min()) * 0.5
    mid_z = (z.max()+z.min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.tight_layout()
    # Save the plot to a file
    plt.savefig('filtered_point_cloud_sampled_visualization.png', dpi=300, bbox_inches='tight')
    print("Plot saved as 'filtered_point_cloud_sampled_visualization.png'")
    plt.show()

if __name__ == "__main__":
    # Load and plot the point cloud
    filename = "new-one-line.txt"
    print(f"Loading point cloud from {filename}...")
    points = load_point_cloud(filename)
    print(f"Loaded {len(points)} points")
    print(f"Original data range - X: [{points[:,0].min():.3f}, {points[:,0].max():.3f}], "
          f"Y: [{points[:,1].min():.3f}, {points[:,1].max():.3f}], "
          f"Z: [{points[:,2].min():.3f}, {points[:,2].max():.3f}]")

    # Filter outliers (remove points with any coordinate > 60)
    print("\nFiltering outliers (max range: 60.0)...")
    filtered_points = filter_outliers(points, max_range=60.0)

    if len(filtered_points) > 0:
        print(f"Filtered data range - X: [{filtered_points[:,0].min():.3f}, {filtered_points[:,0].max():.3f}], "
              f"Y: [{filtered_points[:,1].min():.3f}, {filtered_points[:,1].max():.3f}], "
              f"Z: [{filtered_points[:,2].min():.3f}, {filtered_points[:,2].max():.3f}]")

        # Save full filtered data
        save_filtered_data(filtered_points)

        # Sample 5000 points for visualization to reduce lag
        print("\nSampling 5000 points for visualization...")
        viz_points = uniform_sample_points(filtered_points, num_samples=5000)

        # Plot sampled point cloud
        plot_point_cloud(viz_points)
    else:
        print("No valid points remaining after filtering!")
