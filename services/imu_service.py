"""
IMU Service — loads orientation quaternions and provides yaw lookup by timestamp.

File format (semicolon-delimited, no header):
    timestamp_us ; qx ; qy ; qz ; qw

The yaw is extracted from the quaternion (rotation about the gravity / z-axis)
and can be queried for any timestamp via nearest-neighbour interpolation.
"""

import numpy as np


def _quat_to_yaw(qx, qy, qz, qw):
    """Extract yaw (rotation about z) from a quaternion (x, y, z, w)."""
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.arctan2(siny_cosp, cosy_cosp)


class IMUService:
    """Pre-loads an orientation-quaternion CSV and answers yaw queries."""

    def __init__(self, file_path):
        self.timestamps = []       # microseconds (int64)
        self.yaws = []             # radians

        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split(";")
                if len(parts) < 5:
                    continue
                ts = int(parts[0])
                qx, qy, qz, qw = (float(parts[1]), float(parts[2]),
                                    float(parts[3]), float(parts[4]))
                self.timestamps.append(ts)
                self.yaws.append(_quat_to_yaw(qx, qy, qz, qw))

        self.timestamps = np.array(self.timestamps, dtype=np.int64)
        self.yaws = np.array(self.yaws, dtype=np.float64)

        # Store relative timestamps (microseconds from first reading)
        self._t0 = self.timestamps[0]
        self.rel_timestamps = self.timestamps - self._t0

        print(f"[IMU] Loaded {len(self.yaws)} quaternion readings, "
              f"duration {self.rel_timestamps[-1] / 1e6:.1f}s, "
              f"yaw range [{np.degrees(self.yaws.min()):.1f}°, "
              f"{np.degrees(self.yaws.max()):.1f}°]")

    def yaw_at(self, rel_time_us):
        """Return the yaw (radians) closest to the given relative time (µs).

        Uses nearest-neighbour lookup (fast; IMU rate ≈ lidar rate).
        """
        idx = np.searchsorted(self.rel_timestamps, rel_time_us)
        # Clamp
        idx = np.clip(idx, 0, len(self.rel_timestamps) - 1)
        # Check neighbour on the left
        if idx > 0:
            d_left = abs(self.rel_timestamps[idx - 1] - rel_time_us)
            d_right = abs(self.rel_timestamps[idx] - rel_time_us)
            if d_left < d_right:
                idx = idx - 1
        return self.yaws[idx]

    def delta_yaw(self, rel_time_a_us, rel_time_b_us):
        """Return yaw_b − yaw_a (radians), normalised to (−π, π]."""
        ya = self.yaw_at(rel_time_a_us)
        yb = self.yaw_at(rel_time_b_us)
        d = yb - ya
        # Normalise
        d = (d + np.pi) % (2 * np.pi) - np.pi
        return d
