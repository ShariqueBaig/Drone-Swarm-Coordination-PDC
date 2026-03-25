"""
spatial_grid.py — PDC Project Spring 2026

OPTIMIZATIONS vs original:
────────────────────────────────────────────────────────────────────────────
insert_drones_vectorized():
  BEFORE: Python loop — N calls to int(x // cell_size), int(y // cell_size)
  AFTER:  NumPy floor-division on (N,2) array → (N,) int arrays in 1 op.
          Dict-append loop remains (Python dict limit) but does no math.
  PDC:    MAP skeleton. SIMD: floor-division on whole array = 1 C-level op.
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
from collections import defaultdict


class SpatialGrid:

    def __init__(self, cell_size, width, height):
        self.cell_size = cell_size
        self.width     = width
        self.height    = height
        self.grid      = defaultdict(list)

    def clear(self):
        self.grid.clear()

    def get_cell_coords(self, x, y):
        return (int(x // self.cell_size), int(y // self.cell_size))

    # ── Original loop insert (kept for API compatibility) ─────────────────
    def insert_drones(self, positions, ids):
        for i, pos in enumerate(positions):
            cell = self.get_cell_coords(pos[0], pos[1])
            self.grid[cell].append((i, pos))

    # ── OPTIMIZATION: Vectorized insertion ───────────────────────────────
    # Floor-division for all N positions done in 1 NumPy op.
    # Dict-append loop is O(N) but does zero floating-point work.
    def insert_drones_vectorized(self, positions, ids):
        """
        PDC: MAP skeleton — same hash function applied to each boid position.
        SIMD: single floor-division on (N,2) array replaces N scalar divides.
        """
        cell_x = (positions[:, 0] // self.cell_size).astype(np.int32)
        cell_y = (positions[:, 1] // self.cell_size).astype(np.int32)

        for i in range(len(positions)):
            self.grid[(cell_x[i], cell_y[i])].append((ids[i], positions[i]))

    def get_neighbors(self, pos, radius, include_self=False):
        """Return (drone_idx, dist) pairs within radius of pos."""
        center_cell  = self.get_cell_coords(pos[0], pos[1])
        radius_cells = int(np.ceil(radius / self.cell_size))
        cx, cy       = center_cell
        neighbors    = []

        for dx in range(-radius_cells, radius_cells + 1):
            for dy in range(-radius_cells, radius_cells + 1):
                for drone_idx, drone_pos in self.grid.get((cx+dx, cy+dy), []):
                    if not include_self and np.array_equal(drone_pos, pos):
                        continue
                    dist = np.linalg.norm(drone_pos - pos)
                    if dist <= radius:
                        neighbors.append((drone_idx, dist))

        return neighbors
