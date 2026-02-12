import numpy as np
import pyvista as pv

# ── optional Numba acceleration (50-100× faster ray tracing) ──────────────
try:
    from numba import njit

    @njit(cache=True)
    def _trace_free_rays_nb(ox, oy, hx, hy, obs_idx,
                            log_odds, l_miss, nx, ny):
        """Bresenham free-space ray tracing for all obstacle rays — JIT."""
        for k in range(len(obs_idx)):
            i = obs_idx[k]
            x1 = hx[i]
            y1 = hy[i]
            x = ox
            y = oy
            dx = abs(x1 - x)
            dy = abs(y1 - y)
            if x < x1:
                sx = 1
            else:
                sx = -1
            if y < y1:
                sy = 1
            else:
                sy = -1
            err = dx - dy
            while not (x == x1 and y == y1):
                if 0 <= x < nx and 0 <= y < ny:
                    log_odds[y, x] += l_miss
                e2 = 2 * err
                if e2 > -dy:
                    err -= dy
                    x += sx
                if e2 < dx:
                    err += dx
                    y += sy

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


# Cell states for visualization:
#   unexplored  → 0.5  (gray)
#   free        → 1.0  (white)
#   occupied    → 0.0  (black)

UNEXPLORED = 0.0   # log-odds = 0 → probability 0.5


class OccupancyGrid2D:
    """2-D probabilistic occupancy grid with log-odds ray tracing and
    optional 2.5-D elevation tracking.

    Each cell stores a log-odds value:
        log_odds > 0  →  occupied
        log_odds < 0  →  free
        log_odds = 0  →  unexplored

    When 3-D hit points are supplied (shape (N, 3)) the grid also tracks
    a running-mean elevation (z) per cell, enabling 2.5-D terrain
    visualisation.

    **Ground / obstacle separation** — the caller may pass an
    ``obstacle_mask`` to :meth:`update_scan`.  Only *obstacle* points
    are marked as occupied and trigger free-space ray tracing.  **All**
    valid points (ground + obstacle) contribute to elevation, producing
    a smooth height map even over terrain that is not an obstacle.

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
        elevation_scale=1.0,
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

        # ── 2.5-D elevation tracking ─────────────────────────────────
        self.elevation_sum = np.zeros((self.ny, self.nx), dtype=np.float64)
        self.elevation_count = np.zeros((self.ny, self.nx), dtype=np.int32)
        self.elevation_scale = float(elevation_scale)

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
    def update_scan(self, origin_xy, hit_points, obstacle_mask=None):
        """Trace rays from origin to each hit point.

        Parameters
        ----------
        origin_xy : array-like, shape (2,)
            Sensor position in world frame (x, y).
        hit_points : ndarray, shape (N, 2) or (N, 3)
            Hit positions in world frame.  If three columns are provided
            the third column (z) is used for 2.5-D elevation tracking.
        obstacle_mask : ndarray of bool, shape (N,), optional
            ``True`` for points that are **obstacles** (marked occupied,
            with Bresenham free-space ray tracing).  ``False`` for ground
            points that contribute **only** to the elevation map.
            If *None*, every point is treated as an obstacle (backward
            compatible).
        """
        if hit_points.size == 0:
            return

        n_pts = len(hit_points)
        has_z = hit_points.ndim == 2 and hit_points.shape[1] >= 3

        if obstacle_mask is None:
            obstacle_mask = np.ones(n_pts, dtype=bool)

        ox, oy = self._world_to_grid(origin_xy[0], origin_xy[1])

        # ── batch convert all hit points to grid coords ───────────────
        hx_all, hy_all = self._world_to_grid_batch(
            hit_points[:, 0], hit_points[:, 1],
        )

        valid = (
            (hx_all >= 0) & (hx_all < self.nx) &
            (hy_all >= 0) & (hy_all < self.ny)
        )

        # ── elevation: ALL valid points (ground + obstacle) ───────────
        if has_z and valid.any():
            z_vals = hit_points[valid, 2]
            np.add.at(self.elevation_sum,
                      (hy_all[valid], hx_all[valid]), z_vals)
            np.add.at(self.elevation_count,
                      (hy_all[valid], hx_all[valid]), 1)

        # ── occupancy: obstacle points only ───────────────────────────
        valid_obs = valid & obstacle_mask
        if valid_obs.any():
            np.add.at(self.log_odds,
                      (hy_all[valid_obs], hx_all[valid_obs]), self.l_hit)

        # ── free cells along rays to obstacle endpoints only ──────────
        nx, ny = self.nx, self.ny
        l_miss = self.l_miss
        log_odds = self.log_odds
        obs_idx = np.where(obstacle_mask)[0]

        if len(obs_idx) > 0:
            if _HAS_NUMBA:
                _trace_free_rays_nb(
                    ox, oy, hx_all, hy_all, obs_idx,
                    log_odds, l_miss, nx, ny,
                )
            else:
                for i in obs_idx:
                    free_cells = self._bresenham(
                        ox, oy, int(hx_all[i]), int(hy_all[i]))
                    for fx, fy in free_cells:
                        if 0 <= fx < nx and 0 <= fy < ny:
                            log_odds[fy, fx] += l_miss

        np.clip(log_odds, self.log_odds_min, self.log_odds_max, out=log_odds)

    def reset(self):
        """Zero out all log-odds and elevation (back to unexplored)."""
        self.log_odds[:] = 0.0
        self.elevation_sum[:] = 0.0
        self.elevation_count[:] = 0

    # ------------------------------------------------------------------
    # Probability / display
    # ------------------------------------------------------------------
    def to_probability(self):
        """Convert log-odds to probability [0, 1].
        0.5 = unexplored, <0.5 = free, >0.5 = occupied."""
        return 1.0 / (1.0 + np.exp(-self.log_odds))

    def to_display(self):
        """Map to display values: free=1 (white), unexplored=0.5 (gray), occupied=0 (black)."""
        return 1.0 - self.to_probability()

    # ------------------------------------------------------------------
    # Elevation helpers
    # ------------------------------------------------------------------
    @property
    def elevation(self):
        """Per-cell mean elevation (z).  NaN for unobserved cells."""
        out = np.full((self.ny, self.nx), np.nan, dtype=np.float32)
        mask = self.elevation_count > 0
        out[mask] = (self.elevation_sum[mask]
                     / self.elevation_count[mask]).astype(np.float32)
        return out

    def _vertex_elevation(self):
        """Interpolate cell-centred elevation to vertex positions.

        Returns array of shape (ny+1, nx+1) suitable for StructuredGrid
        vertex z-coordinates.  Unobserved cells contribute z = 0.
        """
        elev = np.nan_to_num(self.elevation, nan=0.0)
        # Pad with edge replication so the 2×2 average yields (ny+1, nx+1)
        padded = np.pad(elev, ((1, 1), (1, 1)), mode='edge')
        vz = 0.25 * (padded[:-1, :-1] + padded[:-1, 1:] +
                      padded[1:, :-1] + padded[1:, 1:])
        return vz * self.elevation_scale

    @property
    def has_elevation(self):
        """True if any cell has received elevation data."""
        return bool(np.any(self.elevation_count > 0))

    def _flat_elevation_data(self):
        """Per-cell mean elevation ravelled in C order (NaN = unobserved)."""
        return self.elevation.ravel(order="C").astype(np.float32)

    # ------------------------------------------------------------------
    # PyVista helpers
    # ------------------------------------------------------------------
    def _flat_cell_data(self):
        return self.to_display().ravel(order="C")

    def create_pyvista_grid(self):
        """Create a PyVista StructuredGrid with 2.5-D elevation support.

        Vertex z-coordinates reflect the tracked terrain elevation
        (scaled by ``elevation_scale``), allowing 3-D visualisation when
        the camera is tilted.

        The grid carries **two** cell-data arrays:
        * ``"occ"``       — display-mapped occupancy (0 = occupied, 1 = free)
        * ``"elevation"`` — per-cell mean z (NaN = unobserved)
        """
        xs = np.linspace(self.min_x,
                         self.min_x + self.nx * self.resolution,
                         self.nx + 1)
        ys = np.linspace(self.min_y,
                         self.min_y + self.ny * self.resolution,
                         self.ny + 1)
        xx, yy = np.meshgrid(xs, ys)
        zz = self._vertex_elevation()

        grid = pv.StructuredGrid()
        grid.dimensions = [self.nx + 1, self.ny + 1, 1]
        grid.points = np.column_stack([xx.ravel(), yy.ravel(),
                                       zz.ravel()])
        grid.cell_data["occ"] = self._flat_cell_data()
        grid.cell_data["elevation"] = self._flat_elevation_data()
        return grid

    def update_pyvista_grid(self, grid):
        """Refresh occupancy, elevation scalars and vertex z-coords."""
        grid.cell_data["occ"] = self._flat_cell_data()
        grid.cell_data["elevation"] = self._flat_elevation_data()
        # Update vertex z for elevation changes
        zz = self._vertex_elevation()
        pts = grid.points.copy()
        pts[:, 2] = zz.ravel()
        grid.points = pts

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def save_csv(self, file_path):
        np.savetxt(file_path, self.to_probability(), delimiter=",")

    def save_npy(self, file_path):
        np.save(file_path, self.to_probability())

    def save_elevation_csv(self, file_path):
        """Save per-cell elevation as CSV (NaN for unobserved cells)."""
        np.savetxt(file_path, self.elevation, delimiter=",")

    def save_elevation_npy(self, file_path):
        """Save per-cell elevation as .npy (NaN for unobserved cells)."""
        np.save(file_path, self.elevation)
