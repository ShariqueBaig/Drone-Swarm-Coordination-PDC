"""
flock3d.py — PDC Project Spring 2026

BUGS FIXED vs original:
────────────────────────────────────────────────────────────────────────────
BUG: self.positions += self.velocities          (original line 117)
FIX: self.positions += self.velocities * config.dt
     Without dt, boids move 250 units/frame instead of 250*0.02 = 5 units.
     At dt=0.02 this is a 50× position jump per frame — boids immediately
     escape to the boundary clamp on the first update.

OPTIMIZATION: Sparse forces via cKDTree (optional, default='sparse')
────────────────────────────────────────────────────────────────────────────
1. Dense (original): O(N²) for all N boids every frame.
2. Sparse (new):     O(N log N + E) using scipy cKDTree.
   PDC: Same reduction as swarm_optimized — T₁ drops from N² to N·k_avg.
   Data skeleton: MAP (same force computation per boid).
   PRAM: CREW (multiple boids read shared neighbor positions/velocities).
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
import config

try:
    from scipy.spatial import cKDTree
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False


class Flock3D:

    def __init__(self, use_sparse=True):
        self.num_boids = config.num_boids
        self.bounds    = np.array([config.width, config.height, config.width], dtype=float)

        self.positions     = np.random.rand(self.num_boids, 3) * self.bounds
        self.velocities    = (np.random.rand(self.num_boids, 3) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 3))
        self.obstacles     = []

        # Use sparse KDTree forces when scipy is available
        self.use_sparse = use_sparse and _HAS_SCIPY

    # ════════════════════════════════════════════════════════════════════════
    # MAIN UPDATE
    # ════════════════════════════════════════════════════════════════════════

    def update(self):
        self.accelerations[:] = 0.0

        if self.use_sparse:
            sep_s, aln_s, coh_s = self._forces_sparse()
        else:
            sep_s, aln_s, coh_s = self._forces_dense()

        # Obstacle avoidance (3D)
        obstacle_steer = np.zeros((self.num_boids, 3))
        if self.obstacles:
            obs_array = np.array(self.obstacles)
            obs_diff  = self.positions[:, np.newaxis, :] - obs_array[np.newaxis, :, :]
            obs_dist  = np.linalg.norm(obs_diff, axis=2)
            obs_mask  = obs_dist < (config.perception_radius + config.obstacle_radius)

            with np.errstate(divide='ignore', invalid='ignore'):
                obs_avoid = (obs_diff / obs_dist[:, :, np.newaxis]) / (
                    obs_dist[:, :, np.newaxis] / config.obstacle_radius)
            obs_avoid[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid, axis=1)
            obstacle_steer = self.steer(obstacle_steer)

        # Boundary avoidance (3D)
        boundary_steer = np.zeros((self.num_boids, 3))
        margin         = config.boundary_margin

        for ax in range(3):
            boundary_steer[self.positions[:, ax] < margin, ax]                    =  config.max_speed
            boundary_steer[self.positions[:, ax] > self.bounds[ax] - margin, ax]  = -config.max_speed
        boundary_steer = self.steer(boundary_steer)

        self.accelerations += sep_s       * config.separation_weight
        self.accelerations += aln_s       * config.alignment_weight
        self.accelerations += coh_s       * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight
        self.accelerations += boundary_steer * config.boundary_weight

        self.velocities += self.accelerations

        speeds     = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (
            self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed

        # ── BUG FIX: was self.positions += self.velocities (missing dt) ──
        # Without dt boids teleport 250 units/frame instead of 5 units/frame
        self.positions += self.velocities * config.dt

        self.positions = np.clip(self.positions, 0, self.bounds)

    # ════════════════════════════════════════════════════════════════════════
    # DENSE FORCES — O(N²), original algorithm
    # ════════════════════════════════════════════════════════════════════════

    def _forces_dense(self):
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        mask = dist_matrix < config.perception_radius

        with np.errstate(divide='ignore', invalid='ignore'):
            sep_vecs = diff_matrix / dist_matrix[:, :, np.newaxis]
        sep_vecs[~mask] = 0
        sep = np.sum(sep_vecs, axis=1)

        neighbor_counts = np.sum(mask, axis=1)[:, np.newaxis]

        aln_sum = np.sum(self.velocities[np.newaxis, :, :] * mask[:, :, np.newaxis], axis=1)
        coh_sum = np.sum(self.positions[np.newaxis,  :, :] * mask[:, :, np.newaxis], axis=1)

        sep_steer = self.steer(sep)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vel = aln_sum / neighbor_counts
        avg_vel[np.isnan(avg_vel)] = 0
        aln_steer = self.steer(avg_vel, subtract_velocity=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_pos = coh_sum / neighbor_counts
        avg_pos[np.isnan(avg_pos)] = 0

        vec_to_com = avg_pos - self.positions
        vec_to_com[neighbor_counts.flatten() == 0] = 0
        coh_steer = self.steer(vec_to_com, subtract_velocity=True)

        return sep_steer, aln_steer, coh_steer

    # ════════════════════════════════════════════════════════════════════════
    # SPARSE FORCES — O(N log N + E) via cKDTree
    # PDC: Reduces T₁ from O(N²) to O(N·k_avg). Same reduction as 2D swarm.
    # ════════════════════════════════════════════════════════════════════════

    def _forces_sparse(self):
        """
        Uses np.add.at scatter over actual neighbor pairs only.
        For N=100, k_avg≈15: E≈750 pairs vs N²=10000 — 13× less work.
        PDC: Brent's theorem: lower T₁ → lower T_p for same p.
        """
        N = self.num_boids

        tree      = cKDTree(self.positions)
        pairs_arr = tree.query_pairs(config.perception_radius, output_type='ndarray')

        sep_f = np.zeros((N, 3))
        aln_f = np.zeros((N, 3))
        coh_f = np.zeros((N, 3))
        nc    = np.zeros(N)

        if len(pairs_arr) == 0:
            return self.steer(sep_f), self.steer(aln_f, True), self.steer(coh_f, True)

        ii = pairs_arr[:, 0].astype(int)
        jj = pairs_arr[:, 1].astype(int)

        diff = self.positions[ii] - self.positions[jj]
        dist = np.linalg.norm(diff, axis=1)

        # Separation
        sep_mask = dist < config.safety_distance
        if np.any(sep_mask):
            v = diff[sep_mask] / np.maximum(dist[sep_mask, np.newaxis], 1e-9)
            np.add.at(sep_f, ii[sep_mask],  v)
            np.add.at(sep_f, jj[sep_mask], -v)

        # Alignment + cohesion
        np.add.at(aln_f, ii, self.velocities[jj])
        np.add.at(aln_f, jj, self.velocities[ii])
        np.add.at(coh_f, ii, self.positions[jj])
        np.add.at(coh_f, jj, self.positions[ii])
        np.add.at(nc, ii, 1)
        np.add.at(nc, jj, 1)

        nc_s  = np.maximum(nc[:, np.newaxis], 1)
        coh_f = coh_f / nc_s - self.positions

        return self.steer(sep_f), self.steer(aln_f / nc_s, True), self.steer(coh_f, True)

    # ════════════════════════════════════════════════════════════════════════
    # STEER + ADD OBSTACLE
    # ════════════════════════════════════════════════════════════════════════

    def steer(self, vectors, subtract_velocity=False):
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid      = magnitudes > 0
        result     = np.zeros_like(vectors)
        result[valid] = (vectors[valid] / magnitudes[valid, np.newaxis]) * config.max_speed

        if subtract_velocity:
            result[valid] -= self.velocities[valid]

        force_mags  = np.linalg.norm(result, axis=1)
        limit_force = force_mags > config.max_force
        result[limit_force] = (
            result[limit_force] / force_mags[limit_force, np.newaxis]) * config.max_force

        return result

    def add_obstacle(self, position):
        self.obstacles.append(position)
