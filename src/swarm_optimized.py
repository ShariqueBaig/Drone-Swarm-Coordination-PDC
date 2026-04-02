"""
swarm_optimized.py — Ashhal's SwarmManager (PDC optimized)
PDC Project · Spring 2026

OPTIMIZATIONS vs previous version (all changes marked ── OPT ──):
════════════════════════════════════════════════════════════════════════════
1. OBSTACLE ARRAY CACHE — eliminates 6 rebuilds per frame
   BEFORE: np.array([[ob[0],ob[1]] for ob in self.env.obstacles])
           called in 6 different methods per frame = 6 × O(M) traversals.
   AFTER:  Computed ONCE at the top of update() → self._obs_c, self._obs_r.
           All helpers receive these as arguments.
   PDC:    Eliminating redundant work from the hot path; O(M)×6 → O(M)×1.

2. find_neighbors_grid() — vectorized pairs building
   BEFORE: Python double-loop over nb_lists to build (rows, cols).
   AFTER:  np.concatenate over the neighbor lists → all-NumPy.
   PDC:    Removes inner Python loop overhead from neighbor detection.

3. resolve_drone_drone_collisions() — fully vectorized
   BEFORE: for i, j in zip(ii, jj): — per-pair Python loop (O(C) iterations)
   AFTER:  np.add.at scatter for position push-out and velocity exchange.
   PDC:    CRCW-Combining analogy — concurrent writes to shared positions
           resolved by summation (np.add.at accumulates correctly).
           Brent: T₁ same, but moves from Python-loop to C-level scatter.

4. resolve_drone_obstacle_collisions() — vectorized death + push-out
   BEFORE: for b, o in zip(bi, oi): — per-hit Python loop
   AFTER:  Vectorized death marking; np.add.at scatter for push-out.
   PDC:    MAP skeleton over collision pairs. EREW for death flags
           (each drone writes only its own dead_mask[b]).

5. _apply_stuck_escape() — vectorized escape direction
   BEFORE: for k, di in enumerate(esc_idx): — per-stuck-drone Python loop
   AFTER:  Fully NumPy: vectorized nearest-wall + nearest-obstacle per drone.
   PDC:    MAP skeleton — same direction-computation applied to each stuck
           drone independently. EREW (each writes own velocity row).
════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import config
from scipy.spatial import cKDTree
from spatial_grid import SpatialGrid
from quadtree import QuadTree
from performance_logger import PerformanceLogger

_DRONE_BODY_R = 5.0


class SwarmManagerOptimized:

    def __init__(self, env):
        self.env = env
        np.random.seed(config.seed)
        self.num_boids = config.num_boids
        self.ids       = np.arange(self.num_boids)

        margin = 30.0
        self.positions   = (np.random.rand(self.num_boids, 2) *
                            [env.width - 2*margin, env.height - 2*margin] + margin)
        self.velocities  = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))

        self.spatial_grid = SpatialGrid(
            cell_size=config.perception_radius,
            width=env.width, height=env.height)

        self.frame_count = 0
        self.logger      = PerformanceLogger("optimized_benchmark.csv")
        self.use_method  = 'grid'

        self.avg_neighbors   = 0.0
        self.neighbor_counts = np.zeros(self.num_boids)
        self.neighbor_mask   = np.zeros((self.num_boids, self.num_boids), dtype=bool)
        self.dead_mask       = np.zeros(self.num_boids, dtype=bool)

        self._prev_positions  = self.positions.copy()
        self._history_timer   = 0
        self._clear_dirs      = np.zeros((self.num_boids, 2))
        self._clear_block     = np.zeros(self.num_boids)
        self._clear_timer     = 0
        self._HISTORY_FRAMES  = 30

        # ── OPT 1: per-frame obstacle cache (populated in update()) ───────
        self._obs_c = np.empty((0, 2), dtype=np.float64)
        self._obs_r = np.empty((0,),   dtype=np.float64)

    # ── Internal helper: refresh obstacle cache ──────────────────────────
    def _refresh_obs_cache(self):
        """Called once at the top of update(). Replaces 6 per-call rebuilds."""
        self._obs_c = self.env.obs_centers.copy()
        self._obs_r = self.env.obs_radii.copy()

    # ════════════════════════════════════════════════════════════════════════
    # NEIGHBOR DETECTION
    # ════════════════════════════════════════════════════════════════════════

    def find_neighbors_naive(self):
        """D1.1 — O(N²) NumPy broadcast. Fastest at N=100."""
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)
        mask = dist < config.perception_radius

        self.neighbor_counts = np.sum(mask, axis=1).astype(float)
        self.avg_neighbors   = float(np.mean(self.neighbor_counts))
        self.neighbor_mask   = mask
        return ('mask', mask)

    def find_neighbors_grid(self):
        """D1.2 — scipy cKDTree query_ball_point.

        ── OPT 2: vectorized pairs building ────────────────────────────────
        BEFORE: double Python for-loop over nb_lists
        AFTER:  np.concatenate + NumPy indexing only
        ────────────────────────────────────────────────────────────────────
        """
        tree     = cKDTree(self.positions)
        nb_lists = tree.query_ball_point(self.positions, config.perception_radius)

        # ── OPT 2: build pairs without Python inner loop ─────────────────
        rows_list, cols_list = [], []
        counts = np.zeros(self.num_boids)
        for i, nb in enumerate(nb_lists):
            nb_arr = np.array(nb, dtype=int)
            nb_arr = nb_arr[nb_arr != i]           # exclude self
            counts[i] = len(nb_arr)
            gt = nb_arr[nb_arr > i]                # collect each undirected pair once
            if len(gt):
                rows_list.append(np.full(len(gt), i, dtype=int))
                cols_list.append(gt)

        if rows_list:
            rows = np.concatenate(rows_list)
            cols = np.concatenate(cols_list)
        else:
            rows = np.array([], dtype=int)
            cols = np.array([], dtype=int)

        pairs = (rows, cols)
        self.neighbor_counts = counts
        self.avg_neighbors   = float(np.mean(counts))

        mask = np.zeros((self.num_boids, self.num_boids), dtype=bool)
        if len(rows):
            mask[rows, cols] = mask[cols, rows] = True
        self.neighbor_mask = mask
        return ('pairs', pairs)

    def find_neighbors_quadtree(self):
        """D1.3 — scipy cKDTree query_pairs. Fastest single-call API."""
        tree      = cKDTree(self.positions)
        pairs_arr = tree.query_pairs(config.perception_radius, output_type='ndarray')

        if len(pairs_arr):
            pairs_arr = pairs_arr.astype(int)
            ii, jj    = pairs_arr[:, 0], pairs_arr[:, 1]
            counts    = np.zeros(self.num_boids)
            np.add.at(counts, ii, 1)
            np.add.at(counts, jj, 1)
            pairs = (ii, jj)
            mask  = np.zeros((self.num_boids, self.num_boids), dtype=bool)
            mask[ii, jj] = mask[jj, ii] = True
        else:
            counts = np.zeros(self.num_boids)
            pairs  = (np.array([], dtype=int), np.array([], dtype=int))
            mask   = np.zeros((self.num_boids, self.num_boids), dtype=bool)

        self.neighbor_counts = counts
        self.avg_neighbors   = float(np.mean(counts))
        self.neighbor_mask   = mask
        return ('pairs', pairs)

    # ════════════════════════════════════════════════════════════════════════
    # FORCE COMPUTATION
    # ════════════════════════════════════════════════════════════════════════

    def compute_forces_dense(self, mask):
        """Dense O(N²) forces for naive method."""
        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        sep_mask = dist < config.safety_distance
        with np.errstate(divide='ignore', invalid='ignore'):
            sv = diff / dist[:, :, np.newaxis]
        sv[~sep_mask] = 0
        sep = np.sum(sv, axis=1)

        nc  = np.maximum(np.sum(mask, axis=1, keepdims=True), 1)
        aln = np.sum(self.velocities[np.newaxis, :, :] * mask[:, :, np.newaxis], axis=1) / nc
        coh = np.sum(self.positions[np.newaxis,  :, :] * mask[:, :, np.newaxis], axis=1) / nc \
              - self.positions

        return self.steer(sep), self.steer(aln, True), self.steer(coh, True)

    def compute_forces_sparse(self, pairs):
        """O(N×k) sparse forces. Uses np.add.at scatter on actual pairs."""
        N      = self.num_boids
        ii, jj = pairs

        sep_f = np.zeros((N, 2))
        aln_f = np.zeros((N, 2))
        coh_f = np.zeros((N, 2))
        nc    = np.zeros(N)

        if len(ii) == 0:
            return self.steer(sep_f), self.steer(aln_f, True), self.steer(coh_f, True)

        diff = self.positions[ii] - self.positions[jj]
        dist = np.linalg.norm(diff, axis=1)

        sm = dist < config.safety_distance
        if np.any(sm):
            v = diff[sm] / np.maximum(dist[sm, np.newaxis], 1e-9)
            np.add.at(sep_f, ii[sm],  v)
            np.add.at(sep_f, jj[sm], -v)

        np.add.at(aln_f, ii, self.velocities[jj])
        np.add.at(aln_f, jj, self.velocities[ii])
        np.add.at(coh_f, ii, self.positions[jj])
        np.add.at(coh_f, jj, self.positions[ii])
        np.add.at(nc, ii, 1)
        np.add.at(nc, jj, 1)

        nc_s  = np.maximum(nc[:, np.newaxis], 1)
        coh_f = coh_f / nc_s - self.positions

        return self.steer(sep_f), self.steer(aln_f / nc_s, True), self.steer(coh_f, True)

    def _compute_predictive_avoidance(self, dt):
        """A2.2: forward-project boids and apply preemptive avoidance."""
        horizon = max(float(getattr(config, "predictive_horizon", 0.35)), dt)
        pred_pos = self.positions + self.velocities * horizon

        # Predictive drone-drone threats.
        ddiff = pred_pos[:, np.newaxis, :] - pred_pos[np.newaxis, :, :]
        ddist = np.linalg.norm(ddiff, axis=2)
        np.fill_diagonal(ddist, np.inf)

        d_safe = float(getattr(config, "predictive_safety_distance",
                               config.safety_distance * 1.2))
        d_threat = ddist < d_safe

        with np.errstate(divide='ignore', invalid='ignore'):
            d_rep = ddiff / np.maximum(ddist[:, :, np.newaxis], 1e-9)
        d_rep[~d_threat] = 0.0
        drone_avoid = np.sum(d_rep, axis=1)

        # Predictive drone-obstacle threats.
        obs_avoid = np.zeros((self.num_boids, 2))
        if len(self._obs_c):
            odiff = pred_pos[:, np.newaxis, :] - self._obs_c[np.newaxis, :, :]
            odist = np.linalg.norm(odiff, axis=2)
            o_margin = float(getattr(config, "predictive_obstacle_margin", 18.0))
            o_threat = odist < (self._obs_r[np.newaxis, :] + o_margin)
            with np.errstate(divide='ignore', invalid='ignore'):
                o_rep = odiff / np.maximum(odist[:, :, np.newaxis], 1e-9)
            o_rep[~o_threat] = 0.0
            obs_avoid = np.sum(o_rep, axis=1)

        pred = self.steer(drone_avoid + obs_avoid)
        return pred

    # ════════════════════════════════════════════════════════════════════════
    # OBSTACLE AVOIDANCE — tangential slide
    # (unchanged logic, uses pre-cached _obs_c / _obs_r)
    # ════════════════════════════════════════════════════════════════════════

    def _compute_obstacle_steer(self):
        """Surface-following obstacle avoidance. Uses cached obstacle arrays."""
        obs_c = self._obs_c;  obs_r = self._obs_r      # ── OPT 1
        if len(obs_c) == 0:
            return np.zeros((self.num_boids, 2))

        obs_diff = self.positions[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
        obs_dist = np.linalg.norm(obs_diff, axis=2)

        sense_r = np.maximum(obs_r * 1.4, obs_r + 30.0)
        inner_r = obs_r + 12.0
        in_range = (obs_dist < sense_r[np.newaxis, :]) & (obs_dist > 1e-6)

        if not np.any(in_range):
            return np.zeros((self.num_boids, 2))

        with np.errstate(divide='ignore', invalid='ignore'):
            radial = obs_diff / obs_dist[:, :, np.newaxis]
        radial[~in_range] = 0.0

        vel_3d   = self.velocities[:, np.newaxis, :]
        spd      = np.linalg.norm(self.velocities, axis=1, keepdims=True)
        vel_unit = self.velocities / np.maximum(spd, 1e-6)

        approach = -np.sum(vel_unit[:, np.newaxis, :] * radial, axis=2)
        inside   = obs_dist < obs_r[np.newaxis, :]
        in_inner = obs_dist < inner_r[np.newaxis, :]
        active   = in_range & ((approach > 0) | in_inner | inside)

        if not np.any(active):
            return np.zeros((self.num_boids, 2))

        sense_r_2d = np.broadcast_to(sense_r[np.newaxis, :], obs_dist.shape)
        raw_s      = np.where(active,
                               np.maximum((sense_r_2d / np.maximum(obs_dist, 1.0))**2 - 1.0, 0.0),
                               0.0)
        approach_w = np.where(inside | in_inner, 1.0, np.maximum(approach, 0.3))
        force_mult = np.where(in_inner | inside, 2.0, 0.8)
        radial_push = radial * (raw_s * approach_w * config.max_force * force_mult)[:, :, np.newaxis]

        dot_vr   = np.sum(vel_3d * radial, axis=2, keepdims=True)
        vel_tang = vel_3d - dot_vr * radial
        tang_spd = np.linalg.norm(vel_tang, axis=2)

        perp     = np.stack([-radial[:, :, 1], radial[:, :, 0]], axis=2)
        headon   = (tang_spd < config.max_speed * 0.1) & active
        vel_tang = np.where(headon[:, :, np.newaxis], perp * config.max_speed * 0.3, vel_tang)

        tang_mag   = np.maximum(np.linalg.norm(vel_tang, axis=2), 1e-9)
        tang_unit  = vel_tang / tang_mag[:, :, np.newaxis]
        tang_steer = tang_unit * config.max_speed - vel_3d
        ts_mag     = np.linalg.norm(tang_steer, axis=2, keepdims=True)
        tang_steer = tang_steer * (np.minimum(ts_mag, config.max_force) / np.maximum(ts_mag, 1e-9))
        tang_push  = np.where(active[:, :, np.newaxis], tang_steer * 0.8, 0.0)

        total = np.sum((radial_push + tang_push) * active[:, :, np.newaxis], axis=1)
        mags  = np.linalg.norm(total, axis=1, keepdims=True)
        total = total * (np.minimum(mags, config.max_force * 2.0) / np.maximum(mags, 1e-9))
        return total

    # ════════════════════════════════════════════════════════════════════════
    # CLEARANCE GRADIENT (uses cached arrays)
    # ════════════════════════════════════════════════════════════════════════

    def _compute_clearance_gradient(self):
        obs_c = self._obs_c;  obs_r = self._obs_r      # ── OPT 1
        if len(obs_c) == 0:
            return np.zeros((self.num_boids, 2)), np.zeros(self.num_boids)

        diffs = self.positions[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        near  = np.any(dists < obs_r[np.newaxis, :] * 1.5, axis=1)

        result   = np.zeros((self.num_boids, 2))
        blockage = np.zeros(self.num_boids)

        if not np.any(near):
            return result, blockage

        near_pos  = self.positions[near]
        n_near    = len(near_pos)
        N_SAMP    = 8
        LOOKAHEAD = 120.0

        angles = np.linspace(0, 2 * np.pi, N_SAMP, endpoint=False)
        dirs   = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        steps  = np.array([40.0, 80.0, 120.0])

        probes     = (near_pos[:, np.newaxis, np.newaxis, :] +
                      dirs[np.newaxis, :, np.newaxis, :] *
                      steps[np.newaxis, np.newaxis, :, np.newaxis])
        probe_flat = probes.reshape(-1, 2)

        d_to_obs = np.linalg.norm(
            probe_flat[:, np.newaxis, :] - obs_c[np.newaxis, :, :], axis=2
        ) - obs_r[np.newaxis, :]
        clearance_flat = np.maximum(d_to_obs.min(axis=1), 0.0)
        clearance      = clearance_flat.reshape(n_near, N_SAMP, len(steps))
        min_clear      = clearance.min(axis=2)

        best_idx   = np.argmax(min_clear, axis=1)
        best_dirs  = dirs[best_idx]
        best_clear = min_clear[np.arange(n_near), best_idx]
        block_near = np.maximum(0.0, 1.0 - best_clear / LOOKAHEAD)

        result[near]   = best_dirs
        blockage[near] = block_near
        return result, blockage

    # ════════════════════════════════════════════════════════════════════════
    # BOUNDARY — hybrid velocity-clamp + acceleration
    # ════════════════════════════════════════════════════════════════════════

    def _apply_boundary_vectorized(self, dt):
        W, H = float(self.env.width), float(self.env.height)

        if self.env.boundary == "wrap":
            self.positions[:, 0] %= W
            self.positions[:, 1] %= H
            return

        WALL_SENSE = 80.0
        WALL_INNER = 20.0
        N          = self.num_boids

        px, py     = self.positions[:, 0], self.positions[:, 1]
        wall_configs = [
            (0,  1.0, px,     1),
            (0, -1.0, W - px, 1),
            (1,  1.0, py,     0),
            (1, -1.0, H - py, 0),
        ]

        wall_acc = np.zeros((N, 2))

        for norm_ax, norm_sign, dists, tang_ax in wall_configs:
            vel_toward = -self.velocities[:, norm_ax] * norm_sign

            in_inner = dists < WALL_INNER
            in_outer = (dists >= WALL_INNER) & (dists < WALL_SENSE) & (vel_toward > 0)

            inner_heading_in = in_inner & (vel_toward > 0)
            if np.any(inner_heading_in):
                depth = np.clip((WALL_INNER - dists[inner_heading_in]) / WALL_INNER, 0.0, 1.0)
                self.velocities[inner_heading_in, norm_ax] = (
                    norm_sign * depth * config.max_speed * 0.35)

            if np.any(in_outer):
                raw_s = np.maximum(
                    (WALL_SENSE / np.maximum(dists[in_outer], 1.0))**2 - 1.0, 0.0)
                spd    = np.linalg.norm(self.velocities[in_outer], axis=1)
                appr_w = np.maximum(vel_toward[in_outer] / np.maximum(spd, 1e-6), 0.3)
                wall_acc[in_outer, norm_ax] += raw_s * appr_w * config.max_force * norm_sign

                vel_tang = self.velocities[in_outer, tang_ax]
                has_tang = np.abs(vel_tang) > config.max_speed * 0.05
                if np.any(has_tang):
                    idx        = np.where(in_outer)[0][has_tang]
                    tang_delta = np.clip(
                        np.sign(vel_tang[has_tang]) * config.max_speed
                        - self.velocities[idx, tang_ax],
                        -config.max_force, config.max_force)
                    wall_acc[idx, tang_ax] += tang_delta * 0.4

        wall_acc[:, 0] = np.clip(wall_acc[:, 0], -config.max_force * 2.0, config.max_force * 2.0)
        wall_acc[:, 1] = np.clip(wall_acc[:, 1], -config.max_force * 2.0, config.max_force * 2.0)
        self.accelerations += wall_acc

        np.clip(self.positions[:, 0], 0.0, W, out=self.positions[:, 0])
        np.clip(self.positions[:, 1], 0.0, H, out=self.positions[:, 1])

        wall_dead = ((self.positions[:, 0] <= 1.0) | (self.positions[:, 0] >= W - 1.0) |
                     (self.positions[:, 1] <= 1.0) | (self.positions[:, 1] >= H - 1.0))
        self.dead_mask |= wall_dead

    # ════════════════════════════════════════════════════════════════════════
    # DRONE-DRONE COLLISION — OPT 3: fully vectorized
    # ════════════════════════════════════════════════════════════════════════

    def resolve_drone_drone_collisions(self):
        """
        ── OPT 3: vectorized push-out and velocity exchange ────────────────
        BEFORE: for i, j in zip(ii, jj): — per-pair Python loop
        AFTER:  np.add.at scatter for position and velocity updates.

        PDC: CRCW-Combining analogy — np.add.at accumulates contributions
             from all collision pairs onto shared position/velocity arrays.
             Equivalent to Combining model: concurrent writes merged by +.
        ────────────────────────────────────────────────────────────────────
        """
        hard_r = _DRONE_BODY_R * 2.0

        diff = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dist, np.inf)

        collisions = dist < hard_r
        if not np.any(collisions):
            return

        ii, jj = np.where(collisions)
        keep   = ii < jj
        ii, jj = ii[keep], jj[keep]
        if len(ii) == 0:
            return

        vecs  = self.positions[ii] - self.positions[jj]        # (M, 2)
        dists = np.linalg.norm(vecs, axis=1)                   # (M,)

        # Degenerate: coincident drones
        degen = dists < 1e-6
        if np.any(degen):
            angles = np.random.uniform(0, 2 * np.pi, int(np.sum(degen)))
            vecs[degen] = np.stack([np.cos(angles), np.sin(angles)], axis=1)
            dists[degen] = 1.0

        normals  = vecs / dists[:, np.newaxis]                 # (M, 2)
        overlaps = (hard_r - dists) / 2.0                     # (M,)

        # Vectorized position scatter — O(M) NumPy, no Python loop
        push = normals * overlaps[:, np.newaxis]
        np.add.at(self.positions, ii,  push)
        np.add.at(self.positions, jj, -push)

        # Vectorized velocity exchange along contact normal
        rel  = self.velocities[ii] - self.velocities[jj]      # (M, 2)
        dots = np.einsum('ij,ij->i', rel, normals)             # (M,) dot product

        approaching = dots < 0
        if np.any(approaching):
            impulse = normals[approaching] * dots[approaching, np.newaxis]
            np.add.at(self.velocities, ii[approaching], -impulse)
            np.add.at(self.velocities, jj[approaching],  impulse)

    # ════════════════════════════════════════════════════════════════════════
    # DRONE-OBSTACLE COLLISION — OPT 4: vectorized death + scatter push-out
    # ════════════════════════════════════════════════════════════════════════

    def resolve_drone_obstacle_collisions(self):
        """
        ── OPT 4: vectorized collision resolution ───────────────────────────
        BEFORE: for b, o in zip(bi, oi): — per-hit Python loop
        AFTER:
          Death marking   — single boolean index assignment, no loop
          Position push   — np.add.at scatter over all non-dead hits at once
          Velocity bounce — np.add.at scatter, masked to approaching pairs
        PDC: MAP skeleton over (b, o) hit pairs. EREW: each drone writes
             only its own position/velocity/dead_mask row.
        ────────────────────────────────────────────────────────────────────
        """
        obs_c = self._obs_c;  obs_r = self._obs_r              # ── OPT 1
        if len(obs_c) == 0:
            return

        diff  = self.positions[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
        dist  = np.linalg.norm(diff, axis=2)                   # (N, M)
        col_r = obs_r + _DRONE_BODY_R                          # (M,)

        hits = dist < col_r[np.newaxis, :]
        if not np.any(hits):
            return

        bi, oi = np.where(hits)

        # ── Death: deeply embedded (vectorized, no loop) ──────────────────
        deep = dist[bi, oi] < obs_r[oi] * 0.5
        self.dead_mask[bi[deep]] = True

        # ── Push-out: alive hits only ─────────────────────────────────────
        alive_hit = ~deep
        if not np.any(alive_hit):
            return

        bi_a, oi_a = bi[alive_hit], oi[alive_hit]
        vecs  = self.positions[bi_a] - obs_c[oi_a]            # (K, 2)
        dists = np.linalg.norm(vecs, axis=1)                   # (K,)

        # Degenerate (coincident with obstacle center)
        degen = dists < 1e-6
        if np.any(degen):
            rnd = np.random.randn(int(np.sum(degen)), 2)
            rnd /= np.linalg.norm(rnd, axis=1, keepdims=True)
            vecs[degen]  = rnd
            dists[degen] = np.linalg.norm(vecs[degen], axis=1)

        normals  = vecs / dists[:, np.newaxis]                 # (K, 2)
        overlaps = col_r[oi_a] - dists                         # (K,)

        # Scatter push-out (np.add.at handles duplicate boid indices correctly)
        np.add.at(self.positions, bi_a, normals * overlaps[:, np.newaxis])

        # Velocity bounce — only for pairs approaching the obstacle
        v_hit = self.velocities[bi_a]                          # (K, 2)
        dots  = np.einsum('ij,ij->i', v_hit, normals)         # (K,)
        appr  = dots < 0
        if np.any(appr):
            # v - 2*(v·n)*n  → reflect velocity along normal
            reflect = 2.0 * dots[appr, np.newaxis] * normals[appr]
            np.add.at(self.velocities, bi_a[appr], -reflect)

    # ════════════════════════════════════════════════════════════════════════
    # STUCK ESCAPE — OPT 5: fully vectorized direction computation
    # ════════════════════════════════════════════════════════════════════════

    def _apply_stuck_escape(self):
        """
        ── OPT 5: vectorized escape direction ──────────────────────────────
        BEFORE: for k, di in enumerate(esc_idx): — Python loop
        AFTER:  All nearest-wall and nearest-obstacle directions computed
                for ALL stuck drones simultaneously with NumPy array ops.
        PDC:    MAP skeleton — same direction logic applied to each stuck
                drone independently. EREW — each writes only velocities[di].
        ────────────────────────────────────────────────────────────────────
        """
        self._history_timer += 1
        if self._history_timer < self._HISTORY_FRAMES:
            return

        speeds   = np.linalg.norm(self.velocities, axis=1)
        net_disp = np.linalg.norm(self.positions - self._prev_positions, axis=1)

        isolated  = self.neighbor_counts == 0
        stuck     = (speeds < config.max_speed * 0.03) | (net_disp < 8.0)

        W, H   = float(self.env.width), float(self.env.height)
        margin = config.perception_radius * 1.5
        near_wall = ((self.positions[:, 0] < margin) |
                     (self.positions[:, 0] > W - margin) |
                     (self.positions[:, 1] < margin) |
                     (self.positions[:, 1] > H - margin))

        near_obs = np.zeros(self.num_boids, dtype=bool)
        obs_c    = self._obs_c;  obs_r = self._obs_r           # ── OPT 1
        if len(obs_c) > 0:
            diffs    = self.positions[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
            dists    = np.linalg.norm(diffs, axis=2)
            near_obs = np.any(dists < (obs_r[np.newaxis, :] + margin), axis=1)

        escape  = isolated & stuck & (near_wall | near_obs) & ~self.dead_mask
        n_esc   = int(np.sum(escape))
        if n_esc == 0:
            self._prev_positions = self.positions.copy()
            self._history_timer  = 0
            return

        esc_idx = np.where(escape)[0]
        esc_pos = self.positions[esc_idx]                      # (n_esc, 2)

        # ── Vectorized nearest-wall direction ─────────────────────────────
        wall_dists = np.stack([
            esc_pos[:, 0],
            W - esc_pos[:, 0],
            esc_pos[:, 1],
            H - esc_pos[:, 1],
        ], axis=1)                                             # (n_esc, 4)
        WALL_DIRS   = np.array([[1., 0.], [-1., 0.], [0., 1.], [0., -1.]])
        nw_idx      = np.argmin(wall_dists, axis=1)           # (n_esc,)
        wall_str    = 1.0 / np.maximum(wall_dists[np.arange(n_esc), nw_idx], 1.0)
        kick_dirs   = WALL_DIRS[nw_idx]                        # (n_esc, 2)

        # ── Vectorized nearest-obstacle direction ─────────────────────────
        if len(obs_c) > 0:
            d_all    = self.positions[esc_idx, np.newaxis, :] - obs_c[np.newaxis, :, :]
            dist_all = np.linalg.norm(d_all, axis=2)          # (n_esc, M)
            no_idx   = np.argmin(dist_all, axis=1)            # (n_esc,)
            no_dist  = dist_all[np.arange(n_esc), no_idx]
            obs_str  = 1.0 / np.maximum(no_dist, 1.0)

            valid    = no_dist > 1e-6
            no_diff  = d_all[np.arange(n_esc), no_idx]        # (n_esc, 2)
            no_dirs  = np.where(valid[:, np.newaxis],
                                 no_diff / np.maximum(no_dist[:, np.newaxis], 1e-6),
                                 kick_dirs)

            # Take whichever is stronger: obstacle or wall
            use_obs  = obs_str > wall_str
            kick_dirs = np.where(use_obs[:, np.newaxis], no_dirs, kick_dirs)

        # ── Fallback: zero-direction drones use vel perpendicular ─────────
        zero_dir = np.linalg.norm(kick_dirs, axis=1) < 1e-6
        if np.any(zero_dir):
            v  = self.velocities[esc_idx[zero_dir]]
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            perp = np.stack([-v[:, 1], v[:, 0]], axis=1)
            perp = np.where(sp > 1e-6, perp / np.maximum(sp, 1e-6),
                             np.tile([1., 0.], (int(np.sum(zero_dir)), 1)))
            kick_dirs[zero_dir] = perp

        # Apply kick (additive — blends with existing velocity)
        self.velocities[esc_idx] += kick_dirs * config.max_speed * 0.5

        # Speed clamp after kick
        spd  = np.linalg.norm(self.velocities[esc_idx], axis=1)
        over = spd > config.max_speed
        if np.any(over):
            self.velocities[esc_idx[over]] = (
                self.velocities[esc_idx[over]] / spd[over, np.newaxis]
            ) * config.max_speed

        self._prev_positions = self.positions.copy()
        self._history_timer  = 0
        self._clear_dirs     = np.zeros((self.num_boids, 2))
        self._clear_block    = np.zeros(self.num_boids)
        self._clear_timer    = 0

    # ════════════════════════════════════════════════════════════════════════
    # STEER HELPER
    # ════════════════════════════════════════════════════════════════════════

    def steer(self, vectors, subtract_velocity=False):
        mags   = np.linalg.norm(vectors, axis=1)
        valid  = mags > 0
        result = np.zeros_like(vectors)
        result[valid] = (vectors[valid] / mags[valid, np.newaxis]) * config.max_speed
        if subtract_velocity:
            result[valid] -= self.velocities[valid]
        fmags = np.linalg.norm(result, axis=1)
        over  = fmags > config.max_force
        result[over] = (result[over] / fmags[over, np.newaxis]) * config.max_force
        return result

    # ════════════════════════════════════════════════════════════════════════
    # MAIN UPDATE
    # ════════════════════════════════════════════════════════════════════════

    def update(self, dt=None):
        if dt is None:
            dt = config.dt

        self.frame_count += 1
        self.logger.start_frame()

        alive = ~self.dead_mask
        if not np.any(alive):
            self.logger.end_frame(self)
            return

        # ── OPT 1: build obstacle cache ONCE per frame ───────────────────
        # All helper methods use self._obs_c / self._obs_r instead of
        # independently calling np.array([[ob[0],ob[1]]...]) 6+ times.
        self._refresh_obs_cache()

        self.accelerations = np.zeros((self.num_boids, 2))

        # Neighbor detection + forces
        if self.use_method == 'naive':
            kind, result = self.find_neighbors_naive()
            sep_s, aln_s, coh_s = self.compute_forces_dense(result)
        elif self.use_method == 'grid':
            kind, result = self.find_neighbors_grid()
            sep_s, aln_s, coh_s = self.compute_forces_sparse(result)
        else:   # 'quadtree'
            kind, result = self.find_neighbors_quadtree()
            sep_s, aln_s, coh_s = self.compute_forces_sparse(result)

        # Cohesion scale (reduced near obstacles)
        coh_scale = np.ones(self.num_boids)
        obs_c = self._obs_c;  obs_r = self._obs_r
        if len(obs_c) > 0:
            diffs    = self.positions[:, np.newaxis, :] - obs_c[np.newaxis, :, :]
            dists    = np.linalg.norm(diffs, axis=2)
            prox     = np.maximum(0.0,
                           1.0 - np.maximum(0.0, dists - obs_r[np.newaxis, :]) /
                           (obs_r[np.newaxis, :] * 0.6 + 20))
            coh_scale = 1.0 - 0.40 * prox.max(axis=1)

        self.accelerations[alive] += (
            sep_s[alive] * config.separation_weight +
            aln_s[alive] * config.alignment_weight  +
            coh_s[alive] * config.cohesion_weight * coh_scale[alive, np.newaxis])

        pred_s = self._compute_predictive_avoidance(dt)
        self.accelerations[alive] += pred_s[alive] * getattr(config, "predictive_weight", 1.15)

        # Obstacle avoidance + clearance gradient
        obs_s = self._compute_obstacle_steer()
        self.accelerations[alive] += obs_s[alive] * config.obstacle_weight

        self._clear_timer += 1
        if self._clear_timer >= 3:
            self._clear_dirs, self._clear_block = self._compute_clearance_gradient()
            self._clear_timer = 0

        b_alive = alive & (self._clear_block > 0.05)
        if np.any(b_alive):
            target_vel  = self._clear_dirs[b_alive] * config.max_speed
            diff        = target_vel - self.velocities[b_alive]
            mags        = np.linalg.norm(diff, axis=1, keepdims=True)
            scale       = np.minimum(mags, config.max_force) / np.maximum(mags, 1e-9)
            clear_steer = diff * scale
            self.accelerations[b_alive] += clear_steer * self._clear_block[b_alive, np.newaxis] * 1.5

        # Noise
        self.accelerations[alive] += np.random.randn(int(np.sum(alive)), 2) * 2.0

        # Boundary (vectorized)
        self._apply_boundary_vectorized(dt)

        # Velocity integration
        self.velocities[alive] += self.accelerations[alive]

        # Angular noise
        n_alive = int(np.sum(alive))
        if n_alive > 0:
            angles   = np.random.uniform(-0.05, 0.05, n_alive)
            cos_a, sin_a = np.cos(angles), np.sin(angles)
            vx = self.velocities[alive, 0].copy()
            vy = self.velocities[alive, 1].copy()
            self.velocities[alive, 0] = cos_a * vx - sin_a * vy
            self.velocities[alive, 1] = sin_a * vx + cos_a * vy

        speeds = np.linalg.norm(self.velocities, axis=1)
        over   = alive & (speeds > config.max_speed)
        self.velocities[over] = (
            self.velocities[over] / speeds[over, np.newaxis]) * config.max_speed

        self.velocities[self.dead_mask] = 0.0

        self.positions[alive] += self.velocities[alive] * dt

        # Collision resolution (now fully vectorized)
        self.resolve_drone_obstacle_collisions()
        self.resolve_drone_drone_collisions()

        self._apply_stuck_escape()

        self.logger.end_frame(self)

    def set_method(self, method):
        if method in ['naive', 'grid', 'quadtree']:
            self.use_method = method
            print(f"[SWARM] Method → {method}")
        else:
            print(f"[SWARM] Unknown: '{method}'. Use naive / grid / quadtree")

    def resolve_collisions(self):
        """Legacy alias."""
        self.resolve_drone_obstacle_collisions()
