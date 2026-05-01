"""
swarm.py -- Decentralized Drone Swarm (2D Version)

PDC Techniques Implemented:
  - Task Parallelism & Fork-Join (ThreadPoolExecutor)
  - Work-Stealing Queue for Coverage Tasks
  - Lock-Free Producer-Consumer Pipeline (for Visualizer decoupling)
  - SIMD / Vectorization (NumPy)
  - Cache-Efficient Spatial Partitioning (Grid Hashing, cKDTree)
  - Pipeline Parallelism (Double Buffering)
"""

import numpy as np
import config
import math
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from scipy.spatial import cKDTree


class SwarmManager:
    def __init__(self, env):
        if hasattr(self, "_pool"):
            self._pool.shutdown(wait=False)

        self.env = env
        np.random.seed(config.seed)
        self.num_boids = config.num_boids
        margin = 50.0

        # Memory Allocation (Contiguous 2D Arrays)
        self.positions = np.ascontiguousarray(
            np.random.rand(self.num_boids, 2) * [env.width, env.height]
        )
        self.velocities = np.ascontiguousarray(
            (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        )
        self.accelerations = np.zeros((self.num_boids, 2))
        self.dead_mask = np.zeros(self.num_boids, dtype=bool)

        # Tasks: (10, 2) — only X, Y
        self.tasks = np.zeros((10, 2))
        self.tasks[0:4] = [
            [env.width * 0.2, env.height * 0.7],
            [env.width * 0.8, env.height * 0.7],
            [env.width * 0.2, env.height * 0.3],
            [env.width * 0.8, env.height * 0.3],
        ]
        for i in range(min(2, len(env.obstacles))):
            self.tasks[4 + i] = env.obstacles[i][:2]
        self.tasks[6] = [env.width * 0.1, env.height * 0.2]
        self.tasks[7] = [env.width * 0.9, env.height * 0.8]
        self.tasks[8] = [env.width * 0.3, env.height * 0.5]   # Pickup Point
        self.tasks[9] = [env.width * 0.7, env.height * 0.5]   # Dropoff Point

        self.assigned_tasks = np.full(self.num_boids, -1, dtype=int)
        self.bids = np.full(self.num_boids, np.inf)
        self.collision_count = 0

        self.mission_type = np.full(self.num_boids, 3, dtype=int)  # Default: Idle (3)
        self.mission_timer = np.random.rand(self.num_boids) * 30.0
        self.transport_phase = np.zeros(self.num_boids, dtype=int)  # 0: Pickup, 1: Dropoff

        # Lightweight mission/consensus metrics for rubric tracking
        self.mission_started_at = {}
        self.mission_pending = {5: False, 6: False, 7: False}
        self.last_mission_completion_time = {}
        self.mission_completed_count = {5: 0, 6: 0, 7: 0}
        self.consensus_updates = 0
        self.tracking_error = 0.0
        self.tracking_error_ema = 0.0

        self.failed_mask = np.zeros(self.num_boids, dtype=bool)
        self.fault_injected = False
        self.frame_count = 0

        # Per-drone targets (2D)
        self.drone_targets = self.positions.copy()
        self.patrol_phase = np.zeros(self.num_boids, dtype=int)

        self.use_method = "naive"
        self._last_pairs_i = np.array([], dtype=int)
        self._last_pairs_j = np.array([], dtype=int)
        self.transport_team_size = int(getattr(config, "transport_team_size", 1))

        self.grid_res = 30
        # 2D coverage grid: (30, 30)
        self.visited_grid = np.zeros((self.grid_res, self.grid_res), dtype=bool)
        self.reserved_grid = np.full((self.grid_res, self.grid_res), -1, dtype=int) # New: tracks which drone is targeting which cell
        self._mark_obstacles_covered()  # New: mark obstacles as already covered
        self.last_grid = np.zeros((self.grid_res, self.grid_res), dtype=bool)

        self._pool = ThreadPoolExecutor(max_workers=config.num_threads)
        self._collision_lock = threading.Lock()

        # Work-Stealing Queue for Coverage
        self._work_queue = deque()
        self._work_queue_lock = threading.Lock()
        self._repopulate_work_queue()

        self._R = float(config.perception_radius)
        self._R_sq = self._R * self._R

        from parallel_metrics import ParallelMetrics
        self.metrics = ParallelMetrics()

        # Lock for async renderer reads
        self.state_lock = threading.Lock()

    # ─────────────────────────────────────────────────────────────────────────
    #  Work-Stealing Queue
    # ─────────────────────────────────────────────────────────────────────────

    def _repopulate_work_queue(self):
        unvisited = np.argwhere(~self.visited_grid)
        if len(unvisited) > 0:
            np.random.shuffle(unvisited)
            with self._work_queue_lock:
                self._work_queue.clear()
                self._work_queue.extend(unvisited.tolist())

    def _steal_work(self):
        with self._work_queue_lock:
            if self._work_queue:
                return self._work_queue.popleft()
        return None

    # ─────────────────────────────────────────────────────────────────────────
    #  Coverage Metric
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def coverage_pct(self):
        visited_count = np.count_nonzero(self.visited_grid)
        return (visited_count / (self.grid_res ** 2)) * 100

    def _mark_obstacles_covered(self):
        """Mark cells containing obstacles as pre-covered (System obstacles only)."""
        obs_pos, obs_rad = self.env.get_obstacle_arrays(exclude_user=True)
        if len(obs_rad) == 0:
            return
        
        cw = self.env.width / self.grid_res
        ch = self.env.height / self.grid_res
        
        x = np.linspace(cw/2, self.env.width - cw/2, self.grid_res)
        y = np.linspace(ch/2, self.env.height - ch/2, self.grid_res)
        xv, yv = np.meshgrid(x, y, indexing='ij')
        cell_centers = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        
        diff = cell_centers[:, np.newaxis, :] - obs_pos[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=-1)
        
        # A cell is covered if its center is within its own radius of the obstacle
        # We use a slight padding to ensure obstacles are fully 'masked'
        is_covered = np.any(dist_sq < (obs_rad + 10.0)**2, axis=1)
        self.visited_grid |= is_covered.reshape(self.grid_res, self.grid_res)

    # ─────────────────────────────────────────────────────────────────────────
    #  Neighbor-Finding Algorithms
    # ─────────────────────────────────────────────────────────────────────────

    def find_neighbors_octree(self):
        """cKDTree — works in any dimension automatically."""
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
        """Grid-hashing spatial partitioning — converted to 2D cells."""
        alive_idx = np.where(~self.dead_mask)[0]
        n = len(alive_idx)
        if n < 2:
            return ([], []), alive_idx

        R_sq, cell = self._R_sq, self._R
        pos = self.positions[alive_idx]  # (n, 2)

        cells = np.floor(pos / cell).astype(int)
        grid_map = {}
        for i, c in enumerate(cells):
            grid_map.setdefault((c[0], c[1]), []).append(i)

        ii_list, jj_list = [], []
        for k, bucket in grid_map.items():
            cx, cy = k
            neighbor_indices = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    b = grid_map.get((cx + dx, cy + dy))
                    if b:
                        neighbor_indices.extend(b)
            if not neighbor_indices:
                continue

            n_idx = np.array(neighbor_indices, dtype=int)
            b_idx = np.array(bucket, dtype=int)
            diff = pos[b_idx][:, np.newaxis, :] - pos[n_idx][np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=-1)

            r, c = np.where(dist_sq < R_sq)
            valid_pairs = b_idx[r] < n_idx[c]

            if np.any(valid_pairs):
                ii_list.extend(alive_idx[b_idx[r][valid_pairs]])
                jj_list.extend(alive_idx[n_idx[c][valid_pairs]])

        if ii_list:
            return (np.asarray(ii_list, dtype=int), np.asarray(jj_list, dtype=int)), alive_idx
        return ([], []), alive_idx

    def find_neighbors_naive(self):
        """Brute-force O(n²) — 2D distances."""
        alive_idx = np.where(~self.dead_mask)[0]
        n = len(alive_idx)
        if n < 2:
            return ([], []), alive_idx
        pos = self.positions[alive_idx]
        diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_sq = np.sum(diff ** 2, axis=-1)
        mask = (dist_sq < self._R_sq) & np.triu(np.ones((n, n), dtype=bool), k=1)
        ii, jj = np.where(mask)
        if len(ii) > 0:
            return (alive_idx[ii], alive_idx[jj]), alive_idx
        return ([], []), alive_idx

    def find_neighbors(self):
        """Dispatch to selected algorithm. Default: naive."""
        if self.use_method == "grid":
            return self.find_neighbors_grid()
        elif self.use_method == "octree":
            return self.find_neighbors_octree()
        else:  # "naive" is the default
            return self.find_neighbors_naive()

    def set_method(self, name):
        if name in ("octree", "grid", "naive"):
            self.use_method = name

    # ─────────────────────────────────────────────────────────────────────────
    #  Steering Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _steer_toward(self, desired, velocities):
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
            steer[over] = (steer[over] / np.maximum(smags[over], 1e-9) * config.max_force)
        result[valid] = steer
        return result

    def steer(self, vectors, subtract_vels=None):
        mags = np.linalg.norm(vectors, axis=1)
        valid = mags > 0
        res = np.zeros_like(vectors)
        res[valid] = (vectors[valid] / np.maximum(mags[valid, np.newaxis], 1e-9)) * config.max_speed

        if subtract_vels is not None:
            if isinstance(subtract_vels, bool) and subtract_vels is True:
                res[valid] -= self.velocities[valid]
            else:
                res[valid] -= subtract_vels[valid]

        fmags = np.linalg.norm(res, axis=1)
        over = fmags > config.max_force
        res[over] = (res[over] / np.maximum(fmags[over, np.newaxis], 1e-9)) * config.max_force
        return res

    # ─────────────────────────────────────────────────────────────────────────
    #  Force Computation (2D)
    # ─────────────────────────────────────────────────────────────────────────

    def compute_forces(self, pairs, alive_idx):
        sep_f = np.zeros((self.num_boids, 2))
        aln_f = np.zeros((self.num_boids, 2))
        coh_f = np.zeros((self.num_boids, 2))
        nc = np.zeros(self.num_boids)

        if len(pairs) == 0 or len(pairs[0]) == 0:
            return sep_f, aln_f, coh_f

        ii, jj = pairs
        diff = self.positions[ii] - self.positions[jj]
        dist = np.linalg.norm(diff, axis=1)

        sm = dist < config.safety_distance
        if np.any(sm):
            v = diff[sm] / np.maximum(dist[sm, np.newaxis], 1e-9)
            weight = (config.safety_distance - dist[sm]) / config.safety_distance
            
            # Boost separation for coverage drones (Type 6) to encourage spreading
            is_cov_i = self.mission_type[ii[sm]] == 6
            is_cov_j = self.mission_type[jj[sm]] == 6
            weight[is_cov_i | is_cov_j] *= 2.0
            
            v *= weight[:, np.newaxis]
            np.add.at(sep_f, ii[sm], v)
            np.add.at(sep_f, jj[sm], -v)

        np.add.at(aln_f, ii, self.velocities[jj])
        np.add.at(aln_f, jj, self.velocities[ii])

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

    # ─────────────────────────────────────────────────────────────────────────
    #  Obstacle Avoidance (2D)
    # ─────────────────────────────────────────────────────────────────────────

    def obstacle_avoidance(self):
        obs_force = np.zeros((self.num_boids, 2))
        obs_pos, obs_rad = self.env.get_obstacle_arrays()  # obs_pos: (M, 2)

        if len(obs_rad) == 0:
            return obs_force

        diff = self.positions[:, np.newaxis, :] - obs_pos[np.newaxis, :, :]
        center_dist = np.linalg.norm(diff, axis=2)

        # Edge-based sensing: react to distance from obstacle boundary, not center.
        edge_dist = center_dist - obs_rad[np.newaxis, :]
        sense_radius = 80.0
        in_range = edge_dist < sense_radius
        if not np.any(in_range):
            return obs_force

        with np.errstate(divide="ignore", invalid="ignore"):
            radial = diff / np.maximum(center_dist[:, :, np.newaxis], 1e-9)

        for m in range(len(obs_rad)):
            active = in_range[:, m]
            if not np.any(active):
                continue

            e = edge_dist[active, m]
            # Strong nonlinear repulsion near boundary and very strong when inside.
            outside = np.maximum(e, 0.0)
            inside = np.maximum(-e, 0.0)
            mag = ((sense_radius - outside) / sense_radius) ** 2 + (inside / np.maximum(obs_rad[m], 1.0)) * 6.0
            obs_force[active] += radial[active, m] * mag[:, np.newaxis] * config.max_force * 2.5

        return self.steer(obs_force)

    # ─────────────────────────────────────────────────────────────────────────
    #  Boundary Avoidance (2D — 4 walls only: X min/max, Y min/max)
    # ─────────────────────────────────────────────────────────────────────────

    def boundary_avoidance(self):
        W, H = self.env.width, self.env.height
        margin = config.boundary_margin
        wall_acc = np.zeros((self.num_boids, 2))

        # Min X
        m_x0 = self.positions[:, 0] < margin
        if np.any(m_x0):
            depth = (margin - self.positions[m_x0, 0]) / margin
            wall_acc[m_x0, 0] += config.max_force * (1.0 + depth * 2.0)

        # Max X
        m_x1 = self.positions[:, 0] > (W - margin)
        if np.any(m_x1):
            depth = (self.positions[m_x1, 0] - (W - margin)) / margin
            wall_acc[m_x1, 0] -= config.max_force * (1.0 + depth * 2.0)

        # Min Y
        m_y0 = self.positions[:, 1] < margin
        if np.any(m_y0):
            depth = (margin - self.positions[m_y0, 1]) / margin
            wall_acc[m_y0, 1] += config.max_force * (1.0 + depth * 2.0)

        # Max Y
        m_y1 = self.positions[:, 1] > (H - margin)
        if np.any(m_y1):
            depth = (self.positions[m_y1, 1] - (H - margin)) / margin
            wall_acc[m_y1, 1] -= config.max_force * (1.0 + depth * 2.0)

        return self.steer(wall_acc)

    # ─────────────────────────────────────────────────────────────────────────
    #  Task Auction
    # ─────────────────────────────────────────────────────────────────────────

    def auction_tasks(self, pairs):
        if len(pairs) == 0 or len(pairs[0]) == 0:
            return
        ii, jj = pairs

        same_m = self.mission_type[ii] == self.mission_type[jj]
        same_t = (self.assigned_tasks[ii] == self.assigned_tasks[jj]) & (self.assigned_tasks[ii] != -1)
        conflict = same_m & same_t

        if np.any(conflict):
            ci, cj = ii[conflict], jj[conflict]
            j_wins = self.bids[cj] < self.bids[ci]
            losers = np.unique(np.concatenate([ci[j_wins], cj[~j_wins]]))
            if len(losers) > 0:
                self.assigned_tasks[losers] = -1
                self.bids[losers] = np.inf

        unassigned = np.where(self.assigned_tasks == -1)[0]
        if len(unassigned) == 0:
            return

        task_ranges = {0: (0, 4), 2: (6, 8), 6: (6, 8), 5: (9, 10), 7: (8, 10)}
        for m_type, (t_start, t_end) in task_ranges.items():
            if m_type in [2, 6]:
                continue  # Coverage drones manage their own drone_targets
            m_mask = self.mission_type[unassigned] == m_type
            m_idx = unassigned[m_mask]
            if len(m_idx) == 0:
                continue
            targets = self.tasks[t_start:t_end]
            if len(targets) == 0:
                continue
            diffs = targets[np.newaxis, :, :] - self.positions[m_idx, np.newaxis, :]
            dists = np.linalg.norm(diffs, axis=2)
            best = np.argmin(dists, axis=1)
            self.assigned_tasks[m_idx] = t_start + best
            self.bids[m_idx] = dists[np.arange(len(m_idx)), best]

    def apply_local_consensus(self, pairs):
        """Lightweight neighbor consensus: lower bid wins and neighbor adopts task."""
        if len(pairs) == 0 or len(pairs[0]) == 0:
            return

        ii, jj = pairs
        same_mission = self.mission_type[ii] == self.mission_type[jj]
        active_pair = (~self.failed_mask[ii]) & (~self.failed_mask[jj])
        valid_pair = same_mission & active_pair
        if not np.any(valid_pair):
            return

        ii_v = ii[valid_pair]
        jj_v = jj[valid_pair]

        assigned_i = self.assigned_tasks[ii_v] != -1
        assigned_j = self.assigned_tasks[jj_v] != -1
        one_missing = assigned_i ^ assigned_j
        if not np.any(one_missing):
            return

        src_i = ii_v[one_missing]
        src_j = jj_v[one_missing]

        i_has = self.assigned_tasks[src_i] != -1
        donor = np.where(i_has, src_i, src_j)
        receiver = np.where(i_has, src_j, src_i)

        self.assigned_tasks[receiver] = self.assigned_tasks[donor]
        self.bids[receiver] = self.bids[donor]
        self.consensus_updates += int(len(receiver))

    def set_fleet_mission(self, mission_id):
        """Central mission command so UI and metrics stay in sync."""
        if mission_id == 7:
            eligible = ~self.dead_mask & ~self.failed_mask
            self.mission_type[eligible] = 3
            self.assigned_tasks[eligible] = -1

            if np.any(eligible):
                pickup = self.tasks[8]
                team_size = max(1, min(self.transport_team_size, int(np.sum(eligible))))
                eligible_idx = np.where(eligible)[0]
                dist = np.linalg.norm(self.positions[eligible_idx] - pickup, axis=1)
                chosen = eligible_idx[np.argsort(dist)[:team_size]]

                self.mission_type[chosen] = 7
                self.assigned_tasks[chosen] = -1
                self.transport_phase[chosen] = 0

            self.mission_started_at[mission_id] = time.time()
            self.mission_pending[mission_id] = True
            return

        alive = ~self.dead_mask
        self.mission_type[alive] = mission_id
        self.assigned_tasks[alive] = -1

        if mission_id in self.mission_pending:
            self.mission_started_at[mission_id] = time.time()
            self.mission_pending[mission_id] = True

    def _update_tracking_error(self, active_idx):
        """Mean distance to assigned/current local objective (lower is better)."""
        if len(active_idx) == 0:
            self.tracking_error = 0.0
            self.tracking_error_ema *= 0.98
            return

        has_task = self.assigned_tasks[active_idx] != -1
        if not np.any(has_task):
            self.tracking_error = 0.0
            self.tracking_error_ema *= 0.98
            return

        idx = active_idx[has_task]
        targets = self.tasks[self.assigned_tasks[idx]].copy()

        is_cov = (self.mission_type[idx] == 2) | (self.mission_type[idx] == 6)
        if np.any(is_cov):
            targets[is_cov] = self.drone_targets[idx[is_cov]]

        is_transport = self.mission_type[idx] == 7
        if np.any(is_transport):
            phase = self.transport_phase[idx[is_transport]]
            t_pick = self.tasks[8]
            t_drop = self.tasks[9]
            targets[is_transport] = np.where(
                phase[:, np.newaxis] == 0,
                t_pick[np.newaxis, :],
                t_drop[np.newaxis, :],
            )

        d = np.linalg.norm(targets - self.positions[idx], axis=1)
        self.tracking_error = float(np.mean(d)) if len(d) > 0 else 0.0
        self.tracking_error_ema = 0.9 * self.tracking_error_ema + 0.1 * self.tracking_error

    def _update_mission_completion_metrics(self):
        now = time.time()

        # Object transport complete once all transport drones have returned to idle
        if self.mission_pending[7] and not np.any(self.mission_type == 7):
            st = self.mission_started_at.get(7)
            if st is not None:
                self.last_mission_completion_time[7] = now - st
                self.mission_completed_count[7] += 1
            self.mission_pending[7] = False

        # Area coverage complete at high map coverage threshold
        if self.mission_pending[6] and self.coverage_pct >= 95.0:
            st = self.mission_started_at.get(6)
            if st is not None:
                self.last_mission_completion_time[6] = now - st
                self.mission_completed_count[6] += 1
            self.mission_pending[6] = False

        # Recall complete when active drones gather near recall target
        if self.mission_pending[5]:
            active_idx = np.where(~self.dead_mask & ~self.failed_mask)[0]
            if len(active_idx) > 0:
                recall_target = self.tasks[9]
                d = np.linalg.norm(self.positions[active_idx] - recall_target, axis=1)
                if np.mean(d) < 80.0:
                    st = self.mission_started_at.get(5)
                    if st is not None:
                        self.last_mission_completion_time[5] = now - st
                        self.mission_completed_count[5] += 1
                    self.mission_pending[5] = False

    # ─────────────────────────────────────────────────────────────────────────
    #  Task Steering (2D)
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_task_steer(self):
        ts = np.zeros_like(self.positions)
        alive = np.where(~self.dead_mask)[0]
        if len(alive) == 0:
            return ts

        self.mission_timer[alive] -= config.dt

        can_timeout = (self.mission_type == 0) | (self.mission_type == 2) | (self.mission_type == 6)
        done = (self.mission_timer <= 0) & can_timeout & (~self.dead_mask)

        if np.any(done):
            self.mission_type[done] = 0
            self.assigned_tasks[done] = -1
            self.mission_timer[done] = np.random.uniform(15, 35, size=np.sum(done))

        mt = self.mission_type[alive]
        mt_eff = mt.copy()
        mt_eff[mt == 6] = 2

        # Idle drones (Type 3): no task force
        idle_mask = (mt_eff == 3)
        if np.any(idle_mask):
            ts[alive[idle_mask]] = 0.0

        tid = self.assigned_tasks[alive]
        has_task = tid != -1

        # ── M0: Seek ──────────────────────────────────────────────────────────
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
                self.mission_type[r_idx] = 6
                self.mission_timer[r_idx] = np.random.uniform(15, 35, size=nr)
                self.assigned_tasks[r_idx] = -1
                self.bids[r_idx] = np.inf

            seeking = ~reached
            if np.any(seeking):
                s_idx = idx0[seeking]
                ts[s_idx] = self._steer_toward(diff0[seeking], self.velocities[s_idx])

        # ── M2 / M6: Coverage ─────────────────────────────────────────────────
        m2 = (mt_eff == 2) & has_task
        if np.any(m2):
            idx2 = alive[m2]
            pos2 = self.positions[idx2]
            # 2D voxel resolution: (2,)
            v_res = np.array([self.env.width / self.grid_res, self.env.height / self.grid_res])

            gxy = np.clip((pos2 / v_res).astype(int), 0, self.grid_res - 1)
            visited = self.visited_grid[gxy[:, 0], gxy[:, 1]]
            timer_low = self.mission_timer[idx2] < 0.8
            needs_new = visited | timer_low

            if np.any(needs_new):
                for k in np.where(needs_new)[0]:
                    i = idx2[k]
                    gx, gy = int(gxy[k, 0]), int(gxy[k, 1])

                    # Clear old reservation
                    old_target = ((self.drone_targets[i] / v_res).astype(int))
                    if 0 <= old_target[0] < self.grid_res and 0 <= old_target[1] < self.grid_res:
                        if self.reserved_grid[old_target[0], old_target[1]] == i:
                            self.reserved_grid[old_target[0], old_target[1]] = -1

                    # 1. Greedy Spatial Search (Nearest unvisited & unreserved)
                    sr = 12 if self.mission_type[i] == 6 else 6
                    rx = slice(max(0, gx - sr), min(self.grid_res, gx + sr))
                    ry = slice(max(0, gy - sr), min(self.grid_res, gy + sr))

                    # Find candidates that are unvisited AND not reserved by others
                    candidates = np.argwhere(~self.visited_grid[rx, ry] & (self.reserved_grid[rx, ry] == -1))
                    
                    if len(candidates) > 0:
                        # Pick nearest to current grid position
                        rel_pos = candidates - [gx - rx.start, gy - ry.start]
                        dists = np.sum(rel_pos**2, axis=1)
                        best_idx = np.argmin(dists)
                        c = candidates[best_idx]
                        tv = np.array([c[0] + rx.start, c[1] + ry.start])
                        
                        # Reserve it
                        self.reserved_grid[tv[0], tv[1]] = i
                    else:
                        # 2. Fallback: Work Stealing
                        stolen = self._steal_work()
                        if stolen is not None:
                            tv = np.array(stolen)
                            self.reserved_grid[tv[0], tv[1]] = i
                        else:
                            tv = np.array([gx, gy])

                    self.drone_targets[i] = (tv + 0.5) * v_res
                    self.mission_timer[i] = 15.0 + np.random.rand() * 10.0

            diff2 = self.drone_targets[idx2] - pos2
            ts[idx2] = self._steer_toward(diff2, self.velocities[idx2])

        # ── M5: Recall (2D tangent orbit) ─────────────────────────────────────
        m5 = (mt_eff == 5) & has_task
        if np.any(m5):
            idx5 = alive[m5]
            diff5 = self.tasks[tid[m5]] - self.positions[idx5]
            dist5 = np.linalg.norm(diff5, axis=1)
            # 2D perpendicular: [-dy, dx]
            tangent = np.column_stack([-diff5[:, 1], diff5[:, 0]])
            tmag = np.linalg.norm(tangent, axis=1, keepdims=True)
            tangent = np.divide(tangent, tmag, out=np.zeros_like(tangent), where=tmag > 1e-9)
            far = dist5 > (80.0 + (idx5 % 20) * 5.0)
            desired5 = tangent * config.max_speed
            if np.any(far):
                desired5[far] = diff5[far] * 0.4 + tangent[far] * config.max_speed * 0.7
            ts[idx5] = self._steer_toward(desired5, self.velocities[idx5])

        # ── M7: Object Transport ──────────────────────────────────────────────
        m7 = (mt_eff == 7) & has_task
        if np.any(m7):
            idx7 = alive[m7]
            # Collective Phase Management
            phase0_idx = idx7[self.transport_phase[idx7] == 0]
            if len(phase0_idx) > 0:
                pickup_pos = self.tasks[8]
                dist_to_pickup = np.linalg.norm(self.positions[phase0_idx] - pickup_pos, axis=1)
                
                # If majority/all drones are close enough, switch all to Phase 1
                if np.mean(dist_to_pickup) < 60.0 or np.all(dist_to_pickup < 100.0):
                    self.transport_phase[idx7] = 1
            
            # Steering logic for each drone based on its current phase
            for i in idx7:
                phase = self.transport_phase[i]
                target_pos = self.tasks[8 if phase == 0 else 9]
                diff = target_pos - self.positions[i]
                dist = np.linalg.norm(diff)

                if phase == 1 and dist < 45.0:
                    # Individual completion check for Phase 1 (Dropoff)
                    # We switch back to idle individually once arrived at dropoff
                    self.mission_type[i] = 3
                    self.assigned_tasks[i] = -1
                    self.bids[i] = np.inf
                    self.transport_phase[i] = 0
                else:
                    ts[i] = self._steer_toward(diff[np.newaxis, :], self.velocities[i, np.newaxis])[0]

        return ts

    # ─────────────────────────────────────────────────────────────────────────
    #  Formation Steering (2D)
    # ─────────────────────────────────────────────────────────────────────────

    def calculate_formation_steer(self):
        fs = np.zeros_like(self.positions)

        doing_formation = (~self.dead_mask) & (self.mission_type == 6)
        if not np.any(doing_formation):
            return fs

        center = np.mean(self.positions[doing_formation], axis=0)
        avg_vel = np.mean(self.velocities[doing_formation], axis=0)
        spd = np.linalg.norm(avg_vel)
        if spd < 1.0:
            return fs  # Don't form grid if barely moving

        dir_vec = avg_vel / spd
        # 2D perpendicular: [-dy, dx]
        side_vec = np.array([-dir_vec[1], dir_vec[0]])
        if np.linalg.norm(side_vec) < 1e-3:
            side_vec = np.array([1.0, 0.0])  # Fallback
        side_vec = side_vec / np.linalg.norm(side_vec)

        row = np.arange(self.num_boids) % 10
        col = np.arange(self.num_boids) // 10
        targets = (
            center[np.newaxis, :]
            - dir_vec[np.newaxis, :] * (row[:, np.newaxis] * 60)
            + side_vec[np.newaxis, :] * (col[:, np.newaxis] * 50 - 250)
        )

        full_steer = self.steer(targets - self.positions, subtract_vels=True)
        fs[doing_formation] = full_steer[doing_formation]
        return fs

    # ─────────────────────────────────────────────────────────────────────────
    #  Failed-Drone Avoidance
    # ─────────────────────────────────────────────────────────────────────────

    def _failed_drone_avoidance(self, active_idx):
        f_obs = np.zeros((self.num_boids, 2))
        if not np.any(self.failed_mask) or len(active_idx) == 0:
            return f_obs
        failed_pos = self.positions[self.failed_mask]
        diff = self.positions[active_idx][:, np.newaxis, :] - failed_pos[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        danger = dist < 15.0
        dc = np.sum(danger, axis=1)
        hd = dc > 0
        if np.any(hd):
            s = np.sum(diff * danger[:, :, np.newaxis], axis=1)
            f_obs[active_idx[hd]] = (s[hd] / dc[hd, np.newaxis]) * 1.5
        return f_obs

    # ─────────────────────────────────────────────────────────────────────────
    #  Fault Injection / Reset / Robustness
    # ─────────────────────────────────────────────────────────────────────────

    def inject_faults(self, percentage=0.15):
        num_fail = int(self.num_boids * percentage)
        potential = np.where(~self.dead_mask & ~self.failed_mask)[0]
        if len(potential) == 0:
            return
        to_fail = np.random.choice(potential, min(num_fail, len(potential)), replace=False)
        self.failed_mask[to_fail] = True
        self.assigned_tasks[to_fail] = -1
        self.bids[to_fail] = np.inf

    def reset_faults(self):
        self.failed_mask[:] = False

    def recall_fleet(self, target=None):
        alive = np.where(~self.dead_mask & ~self.failed_mask)[0]
        if target is None:
            target = np.array([self.env.width / 2, self.env.height / 2])
        self.mission_type[alive] = 5
        self.assigned_tasks[alive] = -1
        self.tasks[9] = target

    def get_robustness_score(self, current_time):
        """
        Robustness = (Coverage / Time) * (Total_Drones / Active_Drones)
        Normalized to a baseline expectation.
        """
        if current_time < 1.0:
            return 1.0
        active = np.sum(~self.dead_mask & ~self.failed_mask)
        if active == 0:
            return 0.0
        efficiency = self.coverage_pct / current_time
        compensation = self.num_boids / active
        score = (efficiency * compensation) * 0.5
        return round(min(score, 5.0), 3)

    # ─────────────────────────────────────────────────────────────────────────
    #  Main Update Loop
    # ─────────────────────────────────────────────────────────────────────────

    def update(self):
        """
        PDC TECHNIQUE: Decoupled Update Loop
        state_lock prevents read tearing by the visualizer.
        """
        self.metrics.start_frame()
        self.frame_count += 1

        self.metrics.start_section("serial_overhead")
        self.env.step(config.dt)
        self.metrics.end_section("serial_overhead")

        # ── 2D Coverage Grid Update ────────────────────────────────────────
        self.metrics.start_section("coverage_grid")
        self.last_grid[:] = self.visited_grid
        alive_mask = ~self.dead_mask
        if np.any(alive_mask):
            gx = np.clip(
                (self.positions[alive_mask, 0] / (self.env.width / self.grid_res)).astype(int),
                0, self.grid_res - 1,
            )
            gy = np.clip(
                (self.positions[alive_mask, 1] / (self.env.height / self.grid_res)).astype(int),
                0, self.grid_res - 1,
            )
            self.visited_grid[gx, gy] = True
            # Clear reservation once visited
            self.reserved_grid[gx, gy] = -1
        
        # New: Re-mark obstacles as covered (handles moving obstacles too)
        self._mark_obstacles_covered()
        # Also clear reservations on obstacles
        self.reserved_grid[self.visited_grid] = -1

        if self.frame_count % 100 == 0:
            self._repopulate_work_queue()
        self.metrics.end_section("coverage_grid")

        alive = ~self.dead_mask
        if not np.any(alive):
            return

        self.accelerations = np.zeros((self.num_boids, 2))

        self.metrics.start_section("neighbor_find")
        pairs, alive_idx = self.find_neighbors()
        self.metrics.end_section("neighbor_find")

        if len(pairs) == 2 and len(pairs[0]) > 0:
            dist = np.linalg.norm(self.positions[pairs[0]] - self.positions[pairs[1]], axis=1)
            new_colls = int(np.sum(dist < 10)) // 2
            with self._collision_lock:
                self.collision_count += new_colls
            self._last_pairs_i, self._last_pairs_j = pairs[0], pairs[1]
        else:
            self._last_pairs_i, self._last_pairs_j = np.array([]), np.array([])

        self.auction_tasks(pairs)
        self.apply_local_consensus(pairs)
        active_idx = np.where(alive & ~self.failed_mask)[0]

        # ── Fork-Join: Parallel Force Computation ─────────────────────────
        self.metrics.start_section("fork_join_dispatch")
        fut_forces = self._pool.submit(self.compute_forces, pairs, active_idx)
        fut_obs    = self._pool.submit(self.obstacle_avoidance)
        fut_form   = self._pool.submit(self.calculate_formation_steer)
        fut_task   = self._pool.submit(self.calculate_task_steer)
        fut_wall   = self._pool.submit(self.boundary_avoidance)
        fut_fail   = self._pool.submit(self._failed_drone_avoidance, active_idx)

        sep_s, aln_s, coh_s = fut_forces.result()
        obs_s  = fut_obs.result()
        form_s = fut_form.result()
        task_s = fut_task.result()
        wall_s = fut_wall.result()
        f_fail = fut_fail.result()
        self.metrics.end_section("fork_join_dispatch")

        self.metrics.start_section("integration")

        with self.state_lock:
            acc = self.accelerations
            acc[active_idx] += sep_s[active_idx]  * config.separation_weight
            acc[active_idx] += aln_s[active_idx]  * config.alignment_weight
            acc[active_idx] += coh_s[active_idx]  * config.cohesion_weight
            acc[active_idx] += obs_s[active_idx]  * config.obstacle_weight
            acc[active_idx] += f_fail[active_idx] * 2.0
            acc[active_idx] += wall_s[active_idx] * config.boundary_weight
            acc[active_idx] += task_s[active_idx] * config.task_weight
            acc[active_idx] += form_s[active_idx] * config.formation_weight

            # Failed drones: zero acceleration (no gravity in 2D)
            failed_idx = np.where(alive & self.failed_mask)[0]
            if len(failed_idx) > 0:
                acc[failed_idx] = 0.0
                self.velocities[failed_idx] *= 0.98

            # Optional waypoint steering
            waypoint_s = np.zeros((self.num_boids, 2))
            if getattr(self.env, "target_waypoint", None) is not None:
                waypoint_s = self.steer(
                    np.array(self.env.target_waypoint) - self.positions,
                    subtract_vels=True,
                )
                acc[active_idx] += waypoint_s[active_idx] * getattr(config, "waypoint_weight", 2.5)

            self.last_sep      = sep_s  * config.separation_weight
            self.last_aln      = aln_s  * config.alignment_weight
            self.last_coh      = coh_s  * config.cohesion_weight
            self.last_waypoint = waypoint_s * getattr(config, "waypoint_weight", 2.5)

            self.velocities[alive] += acc[alive]
            speeds = np.linalg.norm(self.velocities[alive], axis=1)
            over = speeds > config.max_speed
            if np.any(over):
                idx = np.where(alive)[0][over]
                self.velocities[idx] = (
                    self.velocities[idx] / speeds[over, np.newaxis]
                ) * config.max_speed

            self.positions[alive] += self.velocities[alive] * config.dt
            # Clip X and Y only (2D)
            np.clip(self.positions[:, 0], 0, self.env.width,  out=self.positions[:, 0])
            np.clip(self.positions[:, 1], 0, self.env.height, out=self.positions[:, 1])

            # Hard wall response: if a drone reaches a boundary, push it back in
            # and remove the outward velocity component so it cannot stick/clip.
            hit_left = self.positions[:, 0] <= 0
            if np.any(hit_left):
                self.positions[hit_left, 0] = 0.0
                self.velocities[hit_left, 0] = np.abs(self.velocities[hit_left, 0])

            hit_right = self.positions[:, 0] >= self.env.width
            if np.any(hit_right):
                self.positions[hit_right, 0] = float(self.env.width)
                self.velocities[hit_right, 0] = -np.abs(self.velocities[hit_right, 0])

            hit_top = self.positions[:, 1] <= 0
            if np.any(hit_top):
                self.positions[hit_top, 1] = 0.0
                self.velocities[hit_top, 1] = np.abs(self.velocities[hit_top, 1])

            hit_bottom = self.positions[:, 1] >= self.env.height
            if np.any(hit_bottom):
                self.positions[hit_bottom, 1] = float(self.env.height)
                self.velocities[hit_bottom, 1] = -np.abs(self.velocities[hit_bottom, 1])

            # Enforce non-penetration for static/dynamic circular obstacles.
            obs_pos, obs_rad = self.env.get_obstacle_arrays()
            if len(obs_rad) > 0 and len(active_idx) > 0:
                p = self.positions[active_idx]
                diff = p[:, np.newaxis, :] - obs_pos[np.newaxis, :, :]
                dist = np.linalg.norm(diff, axis=2)
                inside = dist < obs_rad[np.newaxis, :]
                if np.any(inside):
                    rows, cols = np.where(inside)
                    for r, c in zip(rows, cols):
                        i = active_idx[r]
                        dv = self.positions[i] - obs_pos[c]
                        d = np.linalg.norm(dv)
                        if d < 1e-9:
                            dv = np.array([1.0, 0.0])
                            d = 1.0
                        n = dv / d
                        self.positions[i] = obs_pos[c] + n * (obs_rad[c] + 1.0)
                        self.velocities[i] -= n * np.dot(self.velocities[i], n)

        self._update_tracking_error(active_idx)
        self._update_mission_completion_metrics()

        self.metrics.end_section("integration")
        self.metrics.end_frame()
