"""
swarm_3d.py -- Decentralized Drone Swarm (M3 Tasks B3.x, D3.x)
Parallel + Vectorized implementation for maximum performance differentiation
between spatial algorithms (Octree / Grid / Naive).

Changes from previous version:
  - calculate_task_steer() fully vectorized (per-mission-type masks, no per-drone loop)
  - auction_tasks() vectorized conflict resolution + per-type batch assignment
  - Failed drone avoidance vectorized (was per-drone loop)
  - Independent force computations parallelized via ThreadPoolExecutor
  - boundary_avoidance() no longer clips positions; clipping moved post-integration
  - _steer_toward() helper for batch steering of arbitrary subsets
"""
import numpy as np
import config
import math
from scipy.spatial import cKDTree
from concurrent.futures import ThreadPoolExecutor


class SwarmManager3D:
    """M3 Tasks (B3.x, D3.x) with Parallel + Vectorized Updates"""

    def __init__(self, env):
        # Shut down old thread pool on re-init (called by simulation3d reset)
        if hasattr(self, "_pool"):
            self._pool.shutdown(wait=False)

        self.env = env
        np.random.seed(config.seed)
        self.num_boids = config.num_boids
        margin = 50.0

        # ── 3D Position / Velocity / Acceleration ─────────────────────────
        self.positions = (
            np.random.rand(self.num_boids, 3)
            * [env.width - 2 * margin, env.height - 2 * margin, env.depth - 2 * margin]
            + margin
        )
        self.velocities = (np.random.rand(self.num_boids, 3) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 3))

        self.dead_mask = np.zeros(self.num_boids, dtype=bool)
        self.fault_injected = False
        self.frame_count = 0

        # ── Task Targets (10 slots) ───────────────────────────────────────
        self.tasks = np.zeros((10, 3))
        self.tasks[0:4] = [
            [env.width * 0.2, env.height * 0.7, env.depth * 0.2],
            [env.width * 0.8, env.height * 0.7, env.depth * 0.2],
            [env.width * 0.2, env.height * 0.7, env.depth * 0.8],
            [env.width * 0.8, env.height * 0.7, env.depth * 0.8],
        ]
        for i in range(min(2, len(env.obstacles))):
            self.tasks[4 + i] = env.obstacles[i][:3]
        self.tasks[6] = [env.width * 0.1, env.height * 0.2, env.depth * 0.1]
        self.tasks[7] = [env.width * 0.9, env.height * 0.8, env.depth * 0.9]

        self.assigned_tasks = np.full(self.num_boids, -1, dtype=int)
        self.bids = np.full(self.num_boids, np.inf)
        self.collision_count = 0

        # ── Mission States ────────────────────────────────────────────────
        # 0=Seeking, 1=Circle, 2=Group Patrol, 3=Swarm, 5=Recall,
        # 6=COVERAGE_PATROL (fast coverage with auction+cohesion)
        coverage_count = max(1, int(self.num_boids * 0.4))
        self.mission_type = np.random.randint(0, 4, self.num_boids)
        self.mission_type[:coverage_count] = 6
        self.mission_timer = np.random.rand(self.num_boids) * 30.0

        # ── M3: Fault Tolerance & Failure States ──────────────────────────
        self.failed_mask = np.zeros(self.num_boids, dtype=bool)
        self.tumble_rot = np.random.rand(self.num_boids, 3) * 360

        self.use_method = "octree"
        self._last_pairs_i = np.array([], dtype=int)
        self._last_pairs_j = np.array([], dtype=int)

        # ── M3: Area Coverage Voxel Grid ──────────────────────────────────
        self.grid_res = 30
        self.visited_grid = np.zeros(
            (self.grid_res, self.grid_res, self.grid_res), dtype=bool
        )
        self.last_grid = np.zeros_like(self.visited_grid)

        # ── Parallelism: Thread pool for concurrent numpy operations ─────
        self._pool = ThreadPoolExecutor(max_workers=4)

        # ── Pre-computed constants ────────────────────────────────────────
        self._R = float(config.perception_radius)
        self._R_sq = self._R * self._R

    # ═══════════════════════════════════════════════════════════════════════
    #  PROPERTIES
    # ═══════════════════════════════════════════════════════════════════════

    @property
    def coverage_pct(self):
        return (np.sum(self.visited_grid) / (self.grid_res ** 3)) * 100

    # ═══════════════════════════════════════════════════════════════════════
    #  NEIGHBOR FINDING — Three clearly differentiated algorithms
    # ═══════════════════════════════════════════════════════════════════════

    def find_neighbors_octree(self):
        """O(N log N) cKDTree — C-optimized spatial indexing (D3.1).
        Fastest: entire tree built and queried in compiled C."""
        alive_idx = np.where(~self.dead_mask)[0]
        if len(alive_idx) < 2:
            return ([], []), alive_idx
        tree = cKDTree(self.positions[alive_idx])
        pairs_arr = tree.query_pairs(self._R, output_type="ndarray")
        if len(pairs_arr) > 0:
            return (
                alive_idx[pairs_arr[:, 0]],
                alive_idx[pairs_arr[:, 1]],
            ), alive_idx
        return ([], []), alive_idx

    def find_neighbors_grid(self):
        """O(N) expected grid-hash neighbor discovery (D2.2 / D3.1).
        Vectorized via grouping to significantly reduce dict overhead."""
        alive_idx = np.where(~self.dead_mask)[0]
        n = len(alive_idx)
        if n < 2:
            return ([], []), alive_idx

        R_sq = self._R_sq
        cell = self._R
        pos = self.positions[alive_idx]

        # Phase 1: Vectorized Hash Map Construction
        cells = np.floor(pos / cell).astype(int)
        
        grid = {}
        for i, c in enumerate(cells):
            grid.setdefault((c[0], c[1], c[2]), []).append(i)

        ii_list, jj_list = [], []
        
        for k, bucket in grid.items():
            cx, cy, cz = k
            
            # Collect all indices in the 27-cell neighborhood
            neighbor_indices = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        b = grid.get((cx + dx, cy + dy, cz + dz))
                        if b:
                            neighbor_indices.extend(b)
                            
            if not neighbor_indices:
                continue
                
            n_idx = np.array(neighbor_indices, dtype=int)
            b_idx = np.array(bucket, dtype=int)
            
            # Vectorized cross-distance computation
            diff = pos[b_idx][:, np.newaxis, :] - pos[n_idx][np.newaxis, :, :]
            dist_sq = np.sum(diff**2, axis=-1)
            
            # Mask valid pairs
            r, c = np.where(dist_sq < R_sq)
            
            # Only keep pairs where b_idx[r] < n_idx[c] to avoid double counting
            valid_pairs = b_idx[r] < n_idx[c]
            
            if np.any(valid_pairs):
                ii_list.extend(alive_idx[b_idx[r][valid_pairs]])
                jj_list.extend(alive_idx[n_idx[c][valid_pairs]])

        if ii_list:
            return (
                np.asarray(ii_list, dtype=int),
                np.asarray(jj_list, dtype=int),
            ), alive_idx
        return ([], []), alive_idx

    def find_neighbors_naive(self):
        """O(N²) brute-force — baseline benchmark (D1.1).
        Vectorized to remove Python loop overhead, strictly math bound."""
        alive_idx = np.where(~self.dead_mask)[0]
        n = len(alive_idx)
        if n < 2:
            return ([], []), alive_idx

        pos = self.positions[alive_idx]
        
        # Pure O(N²) vectorized distance computation
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # Upper triangle to avoid double counting and self-pairs
        mask = (dist_sq < self._R_sq) & np.triu(np.ones((n, n), dtype=bool), k=1)
        ii, jj = np.where(mask)

        if len(ii) > 0:
            return (
                alive_idx[ii],
                alive_idx[jj],
            ), alive_idx
        return ([], []), alive_idx

    def find_neighbors(self):
        """Dispatch to the currently active algorithm."""
        if self.use_method == "grid":
            return self.find_neighbors_grid()
        elif self.use_method == "naive":
            return self.find_neighbors_naive()
        else:  # default: octree (cKDTree)
            return self.find_neighbors_octree()

    def set_method(self, name):
        """Switch algorithm at runtime.  name: 'octree' | 'grid' | 'naive'."""
        if name in ("octree", "grid", "naive"):
            self.use_method = name
            print(f"[SWARM] Algorithm -> {name.upper()}")

    # ═══════════════════════════════════════════════════════════════════════
    #  STEERING HELPERS
    # ═══════════════════════════════════════════════════════════════════════

    def _steer_toward(self, desired, velocities):
        """Vectorized steering for arbitrary (K, 3) subsets.
        Returns (K, 3) limited-force steering vectors."""
        mags = np.linalg.norm(desired, axis=1, keepdims=True)
        valid = (mags > 1e-9).ravel()
        result = np.zeros_like(desired)
        if not np.any(valid):
            return result
        norm_d = desired[valid] / mags[valid] * config.max_speed
        steer = norm_d - velocities[valid]
        smags = np.linalg.norm(steer, axis=1, keepdims=True)
        over = (smags > config.max_force).ravel()
        if np.any(over):
            steer[over] = (
                steer[over] / np.maximum(smags[over], 1e-9) * config.max_force
            )
        result[valid] = steer
        return result

    def steer(self, vectors, subtract_vels=None):
        """Full-array steer (num_boids, 3) — used by compute_forces / formation."""
        mags = np.linalg.norm(vectors, axis=1)
        valid = mags > 0
        res = np.zeros_like(vectors)
        res[valid] = (
            vectors[valid] / np.maximum(mags[valid, np.newaxis], 1e-9)
        ) * config.max_speed

        if subtract_vels is not None:
            if isinstance(subtract_vels, bool) and subtract_vels is True:
                res[valid] -= self.velocities[valid]
            else:
                res[valid] -= subtract_vels[valid]

        fmags = np.linalg.norm(res, axis=1)
        over = fmags > config.max_force
        res[over] = (
            res[over] / np.maximum(fmags[over, np.newaxis], 1e-9)
        ) * config.max_force
        return res

    # ═══════════════════════════════════════════════════════════════════════
    #  BOID FORCE COMPUTATION (already vectorized — kept as-is)
    # ═══════════════════════════════════════════════════════════════════════

    def compute_forces(self, pairs, alive_idx):
        sep_f = np.zeros((self.num_boids, 3))
        aln_f = np.zeros((self.num_boids, 3))
        coh_f = np.zeros((self.num_boids, 3))
        nc = np.zeros(self.num_boids)

        if len(pairs) == 0 or len(pairs[0]) == 0:
            return sep_f, aln_f, coh_f

        ii, jj = pairs
        diff = self.positions[ii] - self.positions[jj]
        dist = np.linalg.norm(diff, axis=1)

        # Separation
        sm = dist < config.safety_distance
        if np.any(sm):
            v = diff[sm] / np.maximum(dist[sm, np.newaxis], 1e-9)
            np.add.at(sep_f, ii[sm], v)
            np.add.at(sep_f, jj[sm], -v)

        # Alignment & Cohesion
        np.add.at(aln_f, ii, self.velocities[jj])
        np.add.at(aln_f, jj, self.velocities[ii])

        # M3 Task C2.2: Mission-based cohesion weighting
        coh_weight_ii = np.ones(len(ii))
        same_mission = self.mission_type[ii] == self.mission_type[jj]
        coh_weight_ii[same_mission] = 2.5

        np.add.at(coh_f, ii, self.positions[jj] * coh_weight_ii[:, np.newaxis])
        np.add.at(coh_f, jj, self.positions[ii] * coh_weight_ii[:, np.newaxis])
        np.add.at(nc, ii, coh_weight_ii)
        np.add.at(nc, jj, coh_weight_ii)

        nc_s = np.maximum(nc[:, np.newaxis], 1)
        return (
            self.steer(sep_f),
            self.steer(aln_f / nc_s, subtract_vels=True),
            self.steer(coh_f / nc_s - self.positions, subtract_vels=True),
        )

    # ═══════════════════════════════════════════════════════════════════════
    #  OBSTACLE AVOIDANCE (already vectorized — kept as-is)
    # ═══════════════════════════════════════════════════════════════════════

    def obstacle_avoidance(self):
        obs_force = np.zeros((self.num_boids, 3))
        if not self.env.all_obstacles:
            return obs_force

        obs = np.array(
            [[o[0], o[1], o[2]] for o in self.env.all_obstacles], dtype=float
        )
        radii = np.array([o[3] for o in self.env.all_obstacles], dtype=float)

        diff = self.positions[:, np.newaxis, :] - obs[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

        in_range = dist < (radii[np.newaxis, :] + 50.0)
        if not np.any(in_range):
            return obs_force

        with np.errstate(divide="ignore", invalid="ignore"):
            radial = diff / np.maximum(dist[:, :, np.newaxis], 1e-9)
        radial[~in_range] = 0.0

        for m in range(len(radii)):
            active = in_range[:, m]
            if np.any(active):
                mag = (radii[m] + 50.0 - dist[active, m]) / 50.0
                obs_force[active] += (
                    radial[active, m] * mag[:, np.newaxis] * config.max_force * 2.0
                )

        return self.steer(obs_force)

    # ═══════════════════════════════════════════════════════════════════════
    #  BOUNDARY AVOIDANCE (force-only; position clipping moved to update)
    # ═══════════════════════════════════════════════════════════════════════

    def boundary_avoidance(self):
        W, H, D = self.env.width, self.env.height, self.env.depth
        margin = config.boundary_margin
        wall_acc = np.zeros((self.num_boids, 3))
        limits = np.array([W, H, D])
        for i in range(3):
            pos_i = self.positions[:, i]
            near_min = pos_i < margin
            wall_acc[near_min, i] += config.max_force
            near_max = pos_i > (limits[i] - margin)
            wall_acc[near_max, i] -= config.max_force
        return self.steer(wall_acc)

    # ═══════════════════════════════════════════════════════════════════════
    #  VECTORIZED AUCTION (was per-pair Python loop)
    # ═══════════════════════════════════════════════════════════════════════

    def auction_tasks(self, pairs):
        """M3 Decentralized auction — vectorized conflict resolution + assignment."""
        if len(pairs) == 0 or len(pairs[0]) == 0:
            return
        ii, jj = pairs

        # ── 1. Conflict Resolution (vectorized) ──────────────────────────
        same_m = self.mission_type[ii] == self.mission_type[jj]
        same_t = (self.assigned_tasks[ii] == self.assigned_tasks[jj]) & (
            self.assigned_tasks[ii] != -1
        )
        conflict = same_m & same_t

        if np.any(conflict):
            ci, cj = ii[conflict], jj[conflict]
            j_wins = self.bids[cj] < self.bids[ci]
            losers = np.unique(np.concatenate([ci[j_wins], cj[~j_wins]]))
            if len(losers) > 0:
                self.assigned_tasks[losers] = -1
                self.bids[losers] = np.inf

        # ── 2. Assignment (vectorized per mission type) ──────────────────
        unassigned = np.where(self.assigned_tasks == -1)[0]
        if len(unassigned) == 0:
            return

        task_ranges = {0: (0, 4), 1: (4, 6), 2: (6, 8), 4: (8, 9), 5: (9, 10)}
        for m_type, (t_start, t_end) in task_ranges.items():
            m_mask = self.mission_type[unassigned] == m_type
            m_idx = unassigned[m_mask]
            if len(m_idx) == 0:
                continue
            targets = self.tasks[t_start:t_end]
            if len(targets) == 0:
                continue
            # (K, T, 3) vectorized distance → batch assignment
            diffs = targets[np.newaxis, :, :] - self.positions[m_idx, np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=2)  # (K, T)
            best = np.argmin(dists, axis=1)
            self.assigned_tasks[m_idx] = t_start + best
            self.bids[m_idx] = dists[np.arange(len(m_idx)), best]

    # ═══════════════════════════════════════════════════════════════════════
    #  VECTORIZED TASK STEERING (was per-drone Python loop — biggest win)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_task_steer(self):
        """Vectorized per-mission-type steering — no per-drone Python loop."""
        ts = np.zeros_like(self.positions)
        alive = np.where(~self.dead_mask)[0]
        if len(alive) == 0:
            return ts

        # Decrement mission timers
        self.mission_timer[alive] -= config.dt

        # Expired timers → reset to Seeking (type 0)
        done = (self.mission_timer <= 0) & (self.mission_type != 0) & (~self.dead_mask)
        if np.any(done):
            self.mission_type[done] = 0
            self.assigned_tasks[done] = -1

        # Working copies
        mt = self.mission_type[alive]
        mt_eff = mt.copy()
        mt_eff[mt == 6] = 2  # Coverage patrol maps to patrol logic
        tid = self.assigned_tasks[alive]
        has_task = tid != -1

        # ── Mission 0: Seeking Hub ────────────────────────────────────────
        m0 = (mt_eff == 0) & has_task
        if np.any(m0):
            idx0 = alive[m0]
            pos0 = self.positions[idx0]
            tgt0 = self.tasks[tid[m0]]
            diff0 = tgt0 - pos0
            dist0 = np.linalg.norm(diff0, axis=1)

            reached = dist0 < 35
            if np.any(reached):
                r_idx = idx0[reached]
                nr = int(np.sum(reached))
                self.mission_type[r_idx] = np.random.choice([1, 2, 4], size=nr)
                self.mission_timer[r_idx] = np.random.uniform(15, 35, size=nr)
                self.assigned_tasks[r_idx] = -1
                self.bids[r_idx] = np.inf

            seeking = ~reached
            if np.any(seeking):
                s_idx = idx0[seeking]
                ts[s_idx] = self._steer_toward(
                    diff0[seeking], self.velocities[s_idx]
                )

        # ── Mission 1: Circle Point ───────────────────────────────────────
        m1 = (mt_eff == 1) & has_task
        if np.any(m1):
            idx1 = alive[m1]
            n1 = len(idx1)
            pos1 = self.positions[idx1]
            tgt1 = self.tasks[tid[m1]]
            diff1 = pos1 - tgt1  # relative to target
            dist1 = np.linalg.norm(diff1, axis=1, keepdims=True)
            radius = 110.0

            # Tangent vector in XZ plane
            tangent = np.column_stack(
                [-diff1[:, 2], np.zeros(n1), diff1[:, 0]]
            )
            tmag = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = np.where(tmag > 1e-9, tangent / tmag, 0)

            correction = ((radius - dist1) / np.maximum(dist1, 1)) * diff1
            desired = tangent * config.max_speed + correction * 1.5
            ts[idx1] = self._steer_toward(desired, self.velocities[idx1])

        # ── Mission 2: Patrol Unvisited Area (includes type 6 coverage) ──
        m2 = (mt_eff == 2) & has_task
        if np.any(m2):
            idx2 = alive[m2]
            pos2 = self.positions[idx2]
            v_res = np.array([
                self.env.width / self.grid_res,
                self.env.height / self.grid_res,
                self.env.depth / self.grid_res,
            ])

            # Grid indices for all patrol drones (vectorized)
            gxyz = np.clip((pos2 / v_res).astype(int), 0, self.grid_res - 1)

            # Check which drones need new targets (vectorized)
            visited = self.visited_grid[gxyz[:, 0], gxyz[:, 1], gxyz[:, 2]]
            timer_low = self.mission_timer[idx2] < 0.8
            needs_new = visited | timer_low

            # Only loop over drones that actually need retargeting
            if np.any(needs_new):
                global_unvisited = None  # lazy computed
                for k in np.where(needs_new)[0]:
                    i = idx2[k]
                    gx, gy, gz = int(gxyz[k, 0]), int(gxyz[k, 1]), int(gxyz[k, 2])
                    sr = 8 if self.mission_type[i] == 6 else 4
                    rx = slice(max(0, gx - sr), min(self.grid_res, gx + sr))
                    ry = slice(max(0, gy - sr), min(self.grid_res, gy + sr))
                    rz = slice(max(0, gz - sr), min(self.grid_res, gz + sr))

                    local_sub = self.visited_grid[rx, ry, rz]
                    lu = np.argwhere(~local_sub)
                    if len(lu) > 0:
                        c = lu[np.random.randint(len(lu))]
                        tv = np.array([
                            c[0] + rx.start, c[1] + ry.start, c[2] + rz.start
                        ])
                    else:
                        if global_unvisited is None:
                            global_unvisited = np.argwhere(~self.visited_grid)
                        if len(global_unvisited) > 0:
                            tv = global_unvisited[
                                np.random.randint(len(global_unvisited))
                            ]
                        else:
                            tv = np.array([gx, gy, gz])

                    self.tasks[6 + (i % 2)] = tv * v_res + (v_res / 2)
                    self.mission_timer[i] = np.random.uniform(5, 10)

            # Steer ALL patrol drones toward their targets (vectorized)
            task_idx = 6 + (idx2 % 2)
            patrol_tgt = self.tasks[task_idx]
            diff2 = patrol_tgt - pos2
            ts[idx2] = self._steer_toward(diff2, self.velocities[idx2])

        # ── Mission 4: Intercept Ghost ────────────────────────────────────
        m4 = (mt_eff == 4) & has_task
        if np.any(m4):
            idx4 = alive[m4]
            pos4 = self.positions[idx4]
            tgt4 = self.tasks[tid[m4]]
            diff4 = tgt4 - pos4
            dist4 = np.linalg.norm(diff4, axis=1)
            safe_r = 40.0

            desired4 = diff4.copy()
            close = dist4 < safe_r
            if np.any(close):
                c_diff = diff4[close]
                nc = int(np.sum(close))
                tangent = np.column_stack(
                    [-c_diff[:, 2], np.zeros(nc), c_diff[:, 0]]
                )
                tmag = np.linalg.norm(tangent, axis=1, keepdims=True)
                tangent = np.where(tmag > 1e-9, tangent / tmag, 0)
                desired4[close] = tangent * config.max_speed

            ts[idx4] = self._steer_toward(desired4, self.velocities[idx4])

        # ── Mission 5: Recall (Orbiting Cloud) ───────────────────────────
        m5 = (mt_eff == 5) & has_task
        if np.any(m5):
            idx5 = alive[m5]
            n5 = len(idx5)
            pos5 = self.positions[idx5]
            tgt5 = self.tasks[tid[m5]]
            diff5 = tgt5 - pos5
            dist5 = np.linalg.norm(diff5, axis=1)

            tangent = np.column_stack(
                [-diff5[:, 2], np.zeros(n5), diff5[:, 0]]
            )
            tmag = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = np.where(tmag > 1e-9, tangent / tmag, 0)

            v_osc = 100 * np.sin(self.frame_count * 0.02 + idx5.astype(float))
            radius_tgt = 80.0 + (idx5 % 20) * 5.0
            far = dist5 > radius_tgt

            desired5 = tangent * config.max_speed  # default: orbit
            if np.any(far):
                desired5[far] = (
                    diff5[far] * 0.4 + tangent[far] * config.max_speed * 0.7
                )

            # Vertical layer oscillation
            desired5[:, 1] += (tgt5[:, 1] + v_osc - pos5[:, 1]) * 0.3

            ts[idx5] = self._steer_toward(desired5, self.velocities[idx5])

        return ts

    # ═══════════════════════════════════════════════════════════════════════
    #  FORMATION STEER (M2 B2.5 — already vectorized, unchanged)
    # ═══════════════════════════════════════════════════════════════════════

    def calculate_formation_steer(self):
        """M2 B2.5: 3D V-Formation Control"""
        fs = np.zeros_like(self.positions)
        alive = ~self.dead_mask
        if not np.any(alive):
            return fs

        center = np.mean(self.positions[alive], axis=0)
        avg_vel = np.mean(self.velocities[alive], axis=0)
        spd = np.linalg.norm(avg_vel)
        if spd < 1e-3:
            return fs

        dir_vec = avg_vel / spd
        side_vec = np.cross(dir_vec, [0, 1, 0])

        row = np.arange(self.num_boids) % 10
        col = np.arange(self.num_boids) // 10

        targets = (
            center[np.newaxis, :]
            - dir_vec[np.newaxis, :] * (row[:, np.newaxis] * 60)
            + side_vec[np.newaxis, :] * (col[:, np.newaxis] * 50 - 250)
        )

        diff = targets - self.positions
        full_steer = self.steer(diff, subtract_vels=True)
        fs[alive] = full_steer[alive]
        return fs

    # ═══════════════════════════════════════════════════════════════════════
    #  VECTORIZED FAILED DRONE AVOIDANCE (was per-drone loop)
    # ═══════════════════════════════════════════════════════════════════════

    def _failed_drone_avoidance(self, active_idx):
        """Vectorized avoidance of failed/falling drones."""
        f_obs = np.zeros((self.num_boids, 3))
        if not np.any(self.failed_mask) or len(active_idx) == 0:
            return f_obs

        failed_pos = self.positions[self.failed_mask]  # (F, 3)
        active_pos = self.positions[active_idx]  # (A, 3)

        # (A, F, 3) pairwise diff
        diff = active_pos[:, np.newaxis, :] - failed_pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)  # (A, F)
        danger = dist < 15.0  # (A, F) bool

        danger_count = np.sum(danger, axis=1)
        has_danger = danger_count > 0
        if np.any(has_danger):
            weighted = diff * danger[:, :, np.newaxis]
            s = np.sum(weighted, axis=1)
            f_obs[active_idx[has_danger]] = (
                s[has_danger] / danger_count[has_danger, np.newaxis]
            ) * 1.5
        return f_obs

    # ═══════════════════════════════════════════════════════════════════════
    #  FLEET MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════

    def recall_fleet(self, target=None):
        """M3 Final: Recall all drones to center (Mission 5 orbiting cloud)."""
        alive = np.where(~self.dead_mask & ~self.failed_mask)[0]
        if len(alive) == 0:
            return
        if target is None:
            target = np.array(
                [self.env.width / 2, self.env.height / 2, self.env.depth / 2]
            )
        self.mission_type[alive] = 5
        self.assigned_tasks[alive] = -1
        self.mission_timer[alive] = 999.0
        if len(self.tasks) < 10:
            new_t = np.zeros((10, 3))
            new_t[: len(self.tasks)] = self.tasks
            self.tasks = new_t
        self.tasks[9] = target
        print(f"[M3 SUCCESS] Mission recall initiated to {target}")

    def check_fault_injection(self):
        # M3 Task A3.3 / B3.3: Randomly fail 20% of drones to test swarm robustness
        # DISABLED: fault injection causing unwanted drone deaths during normal operation
        if False and self.frame_count == 300 and not self.fault_injected:
            kill_count = int(self.num_boids * 0.20)
            indices = np.random.choice(self.num_boids, kill_count, replace=False)
            self.dead_mask[indices] = True
            self.fault_injected = True
            print(f"[FAULT INJECTION] {kill_count} drones have failed unexpectedly!")

    def inject_faults(self, percentage=0.15):
        """Randomly marks drones as 'failed' (B3.x)"""
        num_fail = int(self.num_boids * percentage)
        potential = np.where(~self.dead_mask & ~self.failed_mask)[0]
        if len(potential) == 0:
            return
        to_fail = np.random.choice(
            potential, min(num_fail, len(potential)), replace=False
        )
        self.failed_mask[to_fail] = True
        self.assigned_tasks[to_fail] = -1
        self.bids[to_fail] = np.inf

    def reset_faults(self):
        """Restores all failed drones to active duty"""
        self.failed_mask[:] = False

    # ═══════════════════════════════════════════════════════════════════════
    #  MAIN UPDATE — PARALLELIZED
    #
    #  Numpy operations release the GIL, so ThreadPoolExecutor gives real
    #  concurrency for independent array computations.  The three heavy
    #  force functions (compute_forces, obstacle_avoidance, formation_steer)
    #  run in worker threads while the main thread simultaneously executes
    #  task_steer + boundary_avoidance + failed-drone avoidance.
    # ═══════════════════════════════════════════════════════════════════════

    def update(self):
        self.frame_count += 1
        self.env.step(config.dt)
        self.check_fault_injection()

        # ── Coverage Grid (vectorized bulk update) ────────────────────────
        self.last_grid[:] = self.visited_grid
        alive_mask = ~self.dead_mask
        if np.any(alive_mask):
            pos = self.positions[alive_mask]
            gx = np.clip(
                (pos[:, 0] / (self.env.width / self.grid_res)).astype(int),
                0, self.grid_res - 1,
            )
            gy = np.clip(
                (pos[:, 1] / (self.env.height / self.grid_res)).astype(int),
                0, self.grid_res - 1,
            )
            gz = np.clip(
                (pos[:, 2] / (self.env.depth / self.grid_res)).astype(int),
                0, self.grid_res - 1,
            )
            self.visited_grid[gx, gy, gz] = True

        alive = ~self.dead_mask
        if not np.any(alive):
            return

        self.accelerations = np.zeros((self.num_boids, 3))

        # ── 1. Neighbor Finding (algorithm-dependent bottleneck) ─────────
        pairs, alive_idx = self.find_neighbors()

        # ── 2. Collision Detection (vectorized) ──────────────────────────
        if len(pairs) == 2 and len(pairs[0]) > 0:
            ii, jj = pairs
            dist = np.linalg.norm(
                self.positions[ii] - self.positions[jj], axis=1
            )
            self.collision_count += int(np.sum(dist < 10)) // 2

        # Cache pairs for visualizer
        if len(pairs) == 2 and len(pairs[0]) > 0:
            self._last_pairs_i = np.asarray(pairs[0])
            self._last_pairs_j = np.asarray(pairs[1])
        else:
            self._last_pairs_i = np.array([], dtype=int)
            self._last_pairs_j = np.array([], dtype=int)

        # ── 3. Auction (vectorized) ──────────────────────────────────────
        self.auction_tasks(pairs)

        active_mask = alive & ~self.failed_mask
        active_idx = np.where(active_mask)[0]

        # ── 4. VECTORIZED FORCE COMPUTATION ──────────────────────────────
        # Run synchronously. ThreadPool executor disabled due to GIL locking
        # which throttled lightweight Numpy C-matrices heavily.
        sep_s, aln_s, coh_s = self.compute_forces(pairs, active_idx)
        obs_s = self.obstacle_avoidance()
        form_s = self.calculate_formation_steer()
        
        task_s = self.calculate_task_steer()
        wall_s = self.boundary_avoidance()
        f_fail = self._failed_drone_avoidance(active_idx)

        # ── 5. Accumulate All Forces ─────────────────────────────────────
        acc = self.accelerations
        acc[active_idx] += sep_s[active_idx] * config.separation_weight
        acc[active_idx] += aln_s[active_idx] * config.alignment_weight
        acc[active_idx] += coh_s[active_idx] * config.cohesion_weight
        acc[active_idx] += obs_s[active_idx] * config.obstacle_weight
        acc[active_idx] += f_fail[active_idx] * 2.0
        acc[active_idx] += wall_s[active_idx] * config.boundary_weight
        acc[active_idx] += task_s[active_idx] * config.task_weight
        acc[active_idx] += form_s[active_idx] * config.formation_weight

        # Failed drones: gravity + air resistance
        failed_idx = np.where(alive & self.failed_mask)[0]
        if len(failed_idx) > 0:
            acc[failed_idx] = [0, -18.0, 0]
            self.velocities[failed_idx] *= 0.98

        # Interactive waypoint pull
        waypoint_s = np.zeros((self.num_boids, 3))
        if (
            hasattr(self.env, "target_waypoint")
            and self.env.target_waypoint is not None
        ):
            waypoint_diff = np.array(self.env.target_waypoint) - self.positions
            waypoint_s = self.steer(waypoint_diff, subtract_vels=True)
            wp_weight = getattr(config, "waypoint_weight", 2.5)
            acc[active_idx] += waypoint_s[active_idx] * wp_weight

        # Store for visualizer (C2.4 force breakdown HUD)
        self.last_sep = sep_s * config.separation_weight
        self.last_aln = aln_s * config.alignment_weight
        self.last_coh = coh_s * config.cohesion_weight
        wp_weight = getattr(config, "waypoint_weight", 2.5)
        self.last_waypoint = waypoint_s * wp_weight

        # ── 6. Integrate Velocity + Position ─────────────────────────────
        self.velocities[alive] += acc[alive]

        speeds = np.linalg.norm(self.velocities[alive], axis=1)
        over = speeds > config.max_speed
        if np.any(over):
            idx = np.where(alive)[0][over]
            self.velocities[idx] = (
                self.velocities[idx] / speeds[over, np.newaxis]
            ) * config.max_speed

        self.positions[alive] += self.velocities[alive] * config.dt

        # ── 7. Hard Position Clamp (safety net after integration) ────────
        W, H, D = self.env.width, self.env.height, self.env.depth
        self.positions[:, 0] = np.clip(self.positions[:, 0], 0, W)
        self.positions[:, 1] = np.clip(self.positions[:, 1], 0, H)
        self.positions[:, 2] = np.clip(self.positions[:, 2], 0, D)
