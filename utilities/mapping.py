import numpy as np
import pyvista as pv


# Cell states for visualization:
#   unexplored  → 0.5  (gray)
#   free        → 0.0  (black)
#   occupied    → 1.0  (bright)

UNEXPLORED = 0.0   # log-odds = 0 → probability 0.5


class OccupancyGrid2D:
    """2D probabilistic occupancy grid with log-odds ray tracing.

    Each cell stores a log-odds value:
        log_odds > 0  →  occupied
        log_odds < 0  →  free
        log_odds = 0  →  unexplored

    update_scan() traces a ray from the sensor origin to each hit point,
    marking cells along the ray as free and the endpoint as occupied.

    Internal array layout:
        self.log_odds[iy, ix]   shape = (ny, nx)
    """

    def __init__(
        self,
        min_x, max_x,
        min_y, max_y,
        resolution=0.1,
        p_hit=0.7,
        p_miss=0.4,
        log_odds_min=-5.0,
        log_odds_max=5.0,
    ):
        self.min_x = float(min_x)
        self.max_x = float(max_x)
        self.min_y = float(min_y)
        self.max_y = float(max_y)
        self.resolution = float(resolution)

        self.nx = int(np.ceil((self.max_x - self.min_x) / self.resolution))
        self.ny = int(np.ceil((self.max_y - self.min_y) / self.resolution))

        self.log_odds = np.zeros((self.ny, self.nx), dtype=np.float32)

        self.l_hit = np.log(p_hit / (1.0 - p_hit))
        self.l_miss = np.log(p_miss / (1.0 - p_miss))
        self.log_odds_min = float(log_odds_min)
        self.log_odds_max = float(log_odds_max)

    # ------------------------------------------------------------------
    # Coordinate helpers
    # ------------------------------------------------------------------
    def _world_to_grid(self, wx, wy):
        ix = int(np.floor((wx - self.min_x) / self.resolution))
        iy = int(np.floor((wy - self.min_y) / self.resolution))
        return ix, iy

    def _in_bounds(self, ix, iy):
        return 0 <= ix < self.nx and 0 <= iy < self.ny

    # ------------------------------------------------------------------
    # Bresenham ray trace
    # ------------------------------------------------------------------
    @staticmethod
    def _bresenham(x0, y0, x1, y1):
        """Return list of (ix, iy) cells from (x0,y0) to (x1,y1) exclusive of endpoint."""
        cells = []
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        x, y = x0, y0
        while True:
            if x == x1 and y == y1:
                break
            cells.append((x, y))
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        return cells

    # ------------------------------------------------------------------
    # Batch coordinate conversion
    # ------------------------------------------------------------------
    def _world_to_grid_batch(self, wx, wy):
        """Vectorised world → grid conversion for arrays."""
        ix = np.floor((wx - self.min_x) / self.resolution).astype(int)
        iy = np.floor((wy - self.min_y) / self.resolution).astype(int)
        return ix, iy

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update_scan(self, origin_xy, hit_points):
        """Trace rays from origin to each hit point.

        Parameters
        ----------
        origin_xy : array-like, shape (2,)
            Sensor position in world frame (x, y).
        hit_points : ndarray, shape (N, 2)
            Hit positions in world frame (x, y).
        """
        if hit_points.size == 0:
            return

        ox, oy = self._world_to_grid(origin_xy[0], origin_xy[1])

        # ── batch convert all hit points to grid coords ───────────────
        hx_all, hy_all = self._world_to_grid_batch(
            hit_points[:, 0], hit_points[:, 1],
        )

        # ── batch update occupied cells ───────────────────────────────
        valid = (
            (hx_all >= 0) & (hx_all < self.nx) &
            (hy_all >= 0) & (hy_all < self.ny)
        )
        if valid.any():
            np.add.at(self.log_odds, (hy_all[valid], hx_all[valid]), self.l_hit)

        # ── free cells along each ray (Bresenham) ─────────────────────
        nx, ny = self.nx, self.ny
        l_miss = self.l_miss
        log_odds = self.log_odds
        for i in range(len(hit_points)):
            free_cells = self._bresenham(ox, oy, int(hx_all[i]), int(hy_all[i]))
            for fx, fy in free_cells:
                if 0 <= fx < nx and 0 <= fy < ny:
                    log_odds[fy, fx] += l_miss

        np.clip(log_odds, self.log_odds_min, self.log_odds_max, out=log_odds)

    def reset(self):
        """Zero out all log-odds (back to unexplored)."""
        self.log_odds[:] = 0.0

    # ------------------------------------------------------------------
    # Probability / display
    # ------------------------------------------------------------------
    def to_probability(self):
        """Convert log-odds to probability [0, 1].
        0.5 = unexplored, <0.5 = free, >0.5 = occupied."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def to_display(self):
        """Map to display values: free=0 (black), unexplored=0.5 (gray), occupied=1 (bright)."""
        return self.to_probability()

    # ------------------------------------------------------------------
    # PyVista helpers
    # ------------------------------------------------------------------
    def _flat_cell_data(self):
        return self.to_display().ravel(order="C")

    def create_pyvista_grid(self):
        grid = pv.ImageData(
            dimensions=(self.nx + 1, self.ny + 1, 1),
            spacing=(self.resolution, self.resolution, 1e-6),
            origin=(self.min_x, self.min_y, 0.0),
        )
        grid.cell_data["occ"] = self._flat_cell_data()
        return grid

    def update_pyvista_grid(self, grid):
        grid.cell_data["occ"] = self._flat_cell_data()

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_csv(self, file_path):
        np.savetxt(file_path, self.to_probability(), delimiter=",")

    def save_npy(self, file_path):
        np.save(file_path, self.to_probability())
