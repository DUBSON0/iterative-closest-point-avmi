"""Minimal 2-D pose-graph optimiser (Gauss-Newton on SE(2)).

Nodes are 2-D poses stored as [x, y, θ].
Edges are relative-pose measurements with 3×3 information matrices.

After `optimize()`, read corrected poses via `self.nodes` or
`get_poses_as_matrices()`.
"""

import numpy as np


# ── helpers ──────────────────────────────────────────────────────────────────

def normalize_angle(a):
    """Wrap angle to (-π, π]."""
    return (a + np.pi) % (2 * np.pi) - np.pi


def pose_matrix_to_vec(T):
    """3×3 homogeneous matrix → [x, y, θ]."""
    return np.array([T[0, 2], T[1, 2], np.arctan2(T[1, 0], T[0, 0])])


def pose_vec_to_matrix(v):
    """[x, y, θ] → 3×3 homogeneous matrix."""
    x, y, theta = v
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, x],
                     [s,  c, y],
                     [0,  0, 1]])


def relative_transform_vec(T_i, T_j):
    """Relative transform z_ij = T_i⁻¹ T_j  as [Δx, Δy, Δθ]."""
    T_ij = np.linalg.inv(T_i) @ T_j
    return pose_matrix_to_vec(T_ij)


# ── pose graph ───────────────────────────────────────────────────────────────

class PoseGraph2D:
    """Simple 2-D pose-graph with Gauss-Newton optimisation.

    Usage
    -----
    >>> pg = PoseGraph2D()
    >>> pg.add_node([0, 0, 0])                     # node 0  (fixed)
    >>> pg.add_node([1, 0, 0.1])                    # node 1
    >>> pg.add_edge(0, 1, z_ij, information)        # odometry edge
    >>> pg.add_edge(5, 0, z_lc, information)        # loop closure edge
    >>> pg.optimize()
    >>> corrected = pg.get_poses_as_matrices()
    """

    def __init__(self):
        self.nodes = []   # list of np.ndarray  [x, y, θ]
        self.edges = []   # list of (i, j, z_ij, Ω)

    # ── mutation ─────────────────────────────────────────────────────────

    def add_node(self, pose_vec):
        """Append a pose node.  Returns its index."""
        self.nodes.append(np.asarray(pose_vec, dtype=float).copy())
        return len(self.nodes) - 1

    def add_edge(self, i, j, measurement, information=None):
        """Add a constraint between nodes *i* and *j*.

        Parameters
        ----------
        measurement : array-like, shape (3,)
            Relative transform [Δx, Δy, Δθ] from frame *i* to frame *j*.
        information : array-like, shape (3, 3) or None
            Information (inverse-covariance) matrix.  ``None`` → identity.
        """
        z = np.asarray(measurement, dtype=float).copy()
        omega = np.eye(3) if information is None else np.asarray(information, dtype=float).copy()
        self.edges.append((i, j, z, omega))

    # ── optimisation ─────────────────────────────────────────────────────

    def optimize(self, n_iterations=20, fix_node=0, convergence_eps=1e-6):
        """Run Gauss-Newton to minimise the total edge error.

        The pose at *fix_node* is held constant (anchor).
        """
        n = len(self.nodes)
        if n < 2 or len(self.edges) == 0:
            return

        for iteration in range(n_iterations):
            H = np.zeros((3 * n, 3 * n))
            b = np.zeros(3 * n)

            for (i, j, z_ij, omega) in self.edges:
                e, A, B = self._error_and_jacobians(i, j, z_ij)

                si, sj = 3 * i, 3 * j

                H[si:si+3, si:si+3] += A.T @ omega @ A
                H[si:si+3, sj:sj+3] += A.T @ omega @ B
                H[sj:sj+3, si:si+3] += B.T @ omega @ A
                H[sj:sj+3, sj:sj+3] += B.T @ omega @ B

                b[si:si+3] += A.T @ omega @ e
                b[sj:sj+3] += B.T @ omega @ e

            # Fix the anchor node by adding a large diagonal penalty
            sf = 3 * fix_node
            H[sf:sf+3, :] = 0
            H[:, sf:sf+3] = 0
            H[sf:sf+3, sf:sf+3] = np.eye(3) * 1e10
            b[sf:sf+3] = 0

            # Solve  H Δx = -b
            try:
                dx = np.linalg.solve(H, -b)
            except np.linalg.LinAlgError:
                print(f"  PoseGraph: singular H at iter {iteration}, stopping")
                break

            # Apply update
            for k in range(n):
                self.nodes[k][0] += dx[3*k]
                self.nodes[k][1] += dx[3*k + 1]
                self.nodes[k][2] = normalize_angle(self.nodes[k][2] + dx[3*k + 2])

            step_norm = np.linalg.norm(dx)
            if step_norm < convergence_eps:
                print(f"  PoseGraph converged: iter={iteration}, ||Δx||={step_norm:.2e}")
                break
        else:
            print(f"  PoseGraph max iterations: iter={n_iterations}, ||Δx||={step_norm:.2e}")

    # ── internal ─────────────────────────────────────────────────────────

    def _error_and_jacobians(self, i, j, z_ij):
        """Compute the 3-vector error and 3×3 Jacobians A, B for edge (i, j).

        Error:  e = t2v( Z_ij⁻¹  ·  T_i⁻¹  ·  T_j )

        where t2v extracts [Δx, Δy, Δθ] from a 2-D transform.
        """
        xi, xj = self.nodes[i], self.nodes[j]

        θi = xi[2]
        ci, si = np.cos(θi), np.sin(θi)
        Ri_T = np.array([[ci, si], [-si, ci]])        # R(θi)ᵀ

        dt = xj[:2] - xi[:2]                           # t_j − t_i
        dθ = normalize_angle(xj[2] - xi[2])

        # Predicted relative pose in frame i
        pred_t = Ri_T @ dt
        pred_θ = dθ

        # Error
        e = np.array([
            pred_t[0] - z_ij[0],
            pred_t[1] - z_ij[1],
            normalize_angle(pred_θ - z_ij[2]),
        ])

        # ∂(R_iᵀ · dt)/∂θi
        dRiT_dθ = np.array([[-si, ci], [-ci, -si]])
        dRiT_dt = dRiT_dθ @ dt                        # (2,)

        # Jacobian A = ∂e/∂xi  (3×3)
        A = np.zeros((3, 3))
        A[:2, :2] = -Ri_T
        A[:2,  2] = dRiT_dt
        A[ 2,  2] = -1.0

        # Jacobian B = ∂e/∂xj  (3×3)
        B = np.zeros((3, 3))
        B[:2, :2] = Ri_T
        B[ 2,  2] = 1.0

        return e, A, B

    # ── accessors ────────────────────────────────────────────────────────

    def get_poses_as_matrices(self):
        """Return list of 3×3 homogeneous matrices for every node."""
        return [pose_vec_to_matrix(n) for n in self.nodes]

    def total_error(self):
        """Sum of weighted squared errors across all edges."""
        total = 0.0
        for (i, j, z_ij, omega) in self.edges:
            e, _, _ = self._error_and_jacobians(i, j, z_ij)
            total += e @ omega @ e
        return total
