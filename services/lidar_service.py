import time
import numpy as np


def parse_line_lidar_data(data):
    elements = data.strip().replace(";", " ").split()
    if len(elements) < 2:
        raise ValueError("Invalid lidar line: expected timestamp + values")

    timestamp_raw = int(elements[0])
    float_values = [float(v) for v in elements[1:]]
    if len(float_values) % 3 != 0:
        raise ValueError("Invalid lidar line: values must be x,y,z triples")
    points = np.array(
        [float_values[i:i + 3] for i in range(0, len(float_values), 3)]
    )
    mask = np.all(points == 0, axis=1)
    points = points[~mask]
    return timestamp_raw, points


class LidarService:
    def __init__(self, file_path, sleep_s=0.0, loop=False):
        self.file_path = file_path
        self.sleep_s = sleep_s
        self.loop = loop

    def scans(self):
        """Yield (timestamp_raw, rel_time_us, points) for each scan.

        ``rel_time_us`` is microseconds elapsed since the first scan
        (suitable for IMU time-alignment).
        """
        first_ts = None
        while True:
            with open(self.file_path, "r") as file:
                for line in file:
                    timestamp_raw, points = parse_line_lidar_data(line)
                    if first_ts is None:
                        first_ts = timestamp_raw
                    rel_time_us = timestamp_raw - first_ts
                    yield timestamp_raw, rel_time_us, points
                    if self.sleep_s > 0:
                        time.sleep(self.sleep_s)
            if not self.loop:
                break
