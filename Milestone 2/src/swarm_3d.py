import numpy as np
import config, math
from scipy.spatial import cKDTree

class SwarmManager3D:
    """M3 Tasks (B3.x, D3.x) with Interactive Enhancements"""
    def __init__(self, env):
        self.env = env
        np.random.seed(config.seed)
        self.num_boids = config.num_boids
        margin = 50.0
        
        # Initialize 3D Arrays
        self.positions = (np.random.rand(self.num_boids, 3) * 
                         [env.width - 2*margin, env.height - 2*margin, env.depth - 2*margin] + margin)
        self.velocities = (np.random.rand(self.num_boids, 3) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 3))
        
        self.dead_mask = np.zeros(self.num_boids, dtype=bool)
        self.fault_injected = False
        self.frame_count = 0
        
        # Task 9: RECALL CENTER
        self.tasks = np.zeros((10, 3))
        # Allocators in fixed corners of the sky
        self.tasks[0:4] = [
            [env.width*0.2, env.height*0.7, env.depth*0.2],
            [env.width*0.8, env.height*0.7, env.depth*0.2],
            [env.width*0.2, env.height*0.7, env.depth*0.8],
            [env.width*0.8, env.height*0.7, env.depth*0.8]
        ]
        # Mission Objectives (4-7): Linked to obstacles and corners
        # 4-5: Obstacle Centers
        for i in range(min(2, len(env.obstacles))):
            self.tasks[4+i] = env.obstacles[i][:3]
        # 6-7: Patrol Bounds (Low/High)
        self.tasks[6] = [env.width*0.1, env.height*0.2, env.depth*0.1]
        self.tasks[7] = [env.width*0.9, env.height*0.8, env.depth*0.9]
        
        self.assigned_tasks = np.full(self.num_boids, -1, dtype=int)
        self.bids = np.full(self.num_boids, np.inf)
        self.collision_count = 0

        # Mission States: 0=Seeking, 1=Circle, 2=Group Patrol, 3=Swarm, 5=Recall, 6=COVERAGE_PATROL (fast coverage with auction+cohesion)
        # Spawn first 40% of drones as dedicated fast-coverage team that still use auctions
        coverage_count = max(1, int(self.num_boids * 0.4))  # 40% of drones for fast coverage
        self.mission_type = np.random.randint(0, 4, self.num_boids)
        self.mission_type[:coverage_count] = 6  # Fast coverage patrol team (auctions tasks but prioritizes coverage)
        self.mission_timer = np.random.rand(self.num_boids) * 30.0 
        
        # M3: Fault Tolerance & Failure States
        self.failed_mask = np.zeros(self.num_boids, dtype=bool)
        self.tumble_rot = np.random.rand(self.num_boids, 3) * 360 # Initial visual rotation
        
        self.use_method = 'octree'
        self._last_pairs_i = np.array([], dtype=int)
        self._last_pairs_j = np.array([], dtype=int)

        # M3: Area Coverage Voxel Grid (Brain-managed) - Finer resolution for faster coverage
        self.grid_res = 30  # Higher resolution for faster coverage completion
        self.visited_grid = np.zeros((self.grid_res, self.grid_res, self.grid_res), dtype=bool)
        self.last_grid = np.zeros_like(self.visited_grid) # For discovery tracking

    @property
    def coverage_pct(self):
        return (np.sum(self.visited_grid) / (self.grid_res**3)) * 100

    def recall_fleet(self, target=None):
        """Forces all active drones to return to a specific point (Mission 5)"""
        alive_mask = ~self.dead_mask
        alive_idx = np.where(alive_mask)[0]
        if len(alive_idx) == 0: return
        
        if target is None:
            target = [self.env.width/2, self.env.height/2, self.env.depth/2]
        
        # Override all missions for active drones
        self.mission_type[alive_idx] = 5
        self.assigned_tasks[alive_idx] = 9
        # Ensure task array is sized
        if len(self.tasks) < 10:
            new_t = np.zeros((10,3))
            new_t[:len(self.tasks)] = self.tasks
            self.tasks = new_t
            
        self.tasks[9] = target
        print(f"[Swarm] Mission Success: Recalling {len(alive_idx)} drones to {target}")

    def check_fault_injection(self):
        # M3 Task A3.3 / B3.3: Randomly fail 20% of drones to test swarm robustness
        # DISABLED: fault injection causing unwanted drone deaths during normal operation
        if False and self.frame_count == 300 and not self.fault_injected:
            kill_count = int(self.num_boids * 0.20)
            indices = np.random.choice(self.num_boids, kill_count, replace=False)
            self.dead_mask[indices] = True
            self.fault_injected = True
            print(f"[FAULT INJECTION] {kill_count} drones have failed unexpectedly!")

    def find_neighbors_octree(self):
        # M3 Task D3.1: 3D Spatial Partitioning using cKDTree
        alive_idx = np.where(~self.dead_mask)[0]
        if len(alive_idx) < 2: return [], alive_idx
        
        alive_pos = self.positions[alive_idx]
        tree = cKDTree(alive_pos)
        pairs_arr = tree.query_pairs(config.perception_radius, output_type='ndarray')
        
        if len(pairs_arr) > 0:
            return (alive_idx[pairs_arr[:,0]], alive_idx[pairs_arr[:,1]]), alive_idx
        return ([], []), alive_idx

    def find_neighbors_grid(self):
        """Grid-hash 3D neighbor discovery -- O(N) expected (D2.2 / D3.1)."""
        alive_idx = np.where(~self.dead_mask)[0]
        if len(alive_idx) < 2:
            return ([], []), alive_idx
        R = config.perception_radius
        cell = R
        grid = {}
        for idx in alive_idx:
            p = self.positions[idx]
            k = (int(p[0] // cell), int(p[1] // cell), int(p[2] // cell))
            grid.setdefault(k, []).append(idx)
        ii_list, jj_list = [], []
        for idx in alive_idx:
            p = self.positions[idx]
            cx = int(p[0] // cell)
            cy = int(p[1] // cell)
            cz = int(p[2] // cell)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        for jdx in grid.get((cx+dx, cy+dy, cz+dz), []):
                            if jdx > idx:
                                d = np.linalg.norm(
                                    self.positions[idx] - self.positions[jdx])
                                if d < R:
                                    ii_list.append(idx)
                                    jj_list.append(jdx)
        if ii_list:
            return (np.array(ii_list), np.array(jj_list)), alive_idx
        return ([], []), alive_idx

    def find_neighbors_naive(self):
        """Naive O(N^2) baseline -- for benchmarking comparison (D1.1)."""
        alive_idx = np.where(~self.dead_mask)[0]
        if len(alive_idx) < 2:
            return ([], []), alive_idx
        R = config.perception_radius
        ii_list, jj_list = [], []
        for a_i, idx in enumerate(alive_idx):
            for jdx in alive_idx[a_i + 1:]:
                d = np.linalg.norm(self.positions[idx] - self.positions[jdx])
                if d < R:
                    ii_list.append(idx)
                    jj_list.append(jdx)
        if ii_list:
            return (np.array(ii_list), np.array(jj_list)), alive_idx
        return ([], []), alive_idx

    def find_neighbors(self):
        """Dispatch to the currently active algorithm."""
        if self.use_method == 'grid':
            return self.find_neighbors_grid()
        elif self.use_method == 'naive':
            return self.find_neighbors_naive()
        else:  # default: octree (cKDTree)
            return self.find_neighbors_octree()

    def set_method(self, name):
        """Switch algorithm at runtime. name: 'octree' | 'grid' | 'naive'."""
        if name in ('octree', 'grid', 'naive'):
            self.use_method = name
            print(f'[SWARM] Algorithm -> {name.upper()}')
        
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

        # Alignment & Cohesion Vectorization
        np.add.at(aln_f, ii, self.velocities[jj])
        np.add.at(aln_f, jj, self.velocities[ii])
        
        # M3 Task C2.2: Task-Specific Cohesion (Drones cluster by mission)
        coh_weight_ii = np.ones(len(ii))
        same_mission = self.mission_type[ii] == self.mission_type[jj]
        coh_weight_ii[same_mission] = 2.5 # Stronger pull for group members
        
        np.add.at(coh_f, ii, self.positions[jj] * coh_weight_ii[:, np.newaxis])
        np.add.at(coh_f, jj, self.positions[ii] * coh_weight_ii[:, np.newaxis])
        
        np.add.at(nc, ii, coh_weight_ii)
        np.add.at(nc, jj, coh_weight_ii)

        nc_s = np.maximum(nc[:, np.newaxis], 1)
        return self.steer(sep_f), self.steer(aln_f/nc_s, subtract_vels=True), self.steer(coh_f/nc_s - self.positions, subtract_vels=True)

    def obstacle_avoidance(self):
        obs_force = np.zeros((self.num_boids, 3))
        if not self.env.all_obstacles: return obs_force

        obs = np.array([[o[0], o[1], o[2]] for o in self.env.all_obstacles], dtype=float)
        radii = np.array([o[3] for o in self.env.all_obstacles], dtype=float)

        diff = self.positions[:, np.newaxis, :] - obs[np.newaxis, :, :] 
        dist = np.linalg.norm(diff, axis=2)

        in_range = dist < (radii[np.newaxis, :] + 50.0) 
        if not np.any(in_range): return obs_force

        with np.errstate(divide='ignore', invalid='ignore'):
            radial = diff / np.maximum(dist[:, :, np.newaxis], 1e-9)
            
        radial[~in_range] = 0.0

        for m in range(len(radii)):
            active = in_range[:, m]
            if np.any(active):
                mag = (radii[m] + 50.0 - dist[active, m]) / 50.0
                obs_force[active] += radial[active, m] * mag[:, np.newaxis] * config.max_force * 2.0

        return self.steer(obs_force)

    def boundary_avoidance(self):
        W, H, D = self.env.width, self.env.height, self.env.depth
        margin = config.boundary_margin
        
        wall_acc = np.zeros((self.num_boids, 3))
        for i in range(3):
            limit = [W, H, D][i]
            pos = self.positions[:, i]
            near_min = pos < margin
            wall_acc[near_min, i] += config.max_force
            near_max = pos > (limit - margin)
            wall_acc[near_max, i] -= config.max_force
            
        self.positions[:,0] = np.clip(self.positions[:,0], 0, W)
        self.positions[:,1] = np.clip(self.positions[:,1], 0, H)
        self.positions[:,2] = np.clip(self.positions[:,2], 0, D)
        return self.steer(wall_acc)

    def auction_tasks(self, pairs):
        """M3 Multi-Phase Auctioning: Conflicts resolved within the same Mission Fleet"""
        if len(pairs) == 0 or len(pairs[0]) == 0: return
        ii, jj = pairs
        
        # 1. Conflict Resolution (Neighbors Yield)
        for i, j in zip(ii, jj):
            # Only auction against others in the SAME MISSION
            if self.mission_type[i] == self.mission_type[j]:
                task_i = self.assigned_tasks[i]
                if task_i != -1 and self.assigned_tasks[j] == task_i:
                    if self.bids[j] < self.bids[i]:
                        self.assigned_tasks[i] = -1
                        self.bids[i] = np.inf
                    elif self.bids[i] < self.bids[j]:
                        self.assigned_tasks[j] = -1
                        self.bids[j] = np.inf

        # 2. Assignment (Unassigned drones pick the best task for their mission type)
        unassigned = self.assigned_tasks == -1
        for i in np.where(unassigned)[0]:
            m_type = self.mission_type[i]
            # Select valid task sub-range based on mission phase
            if m_type == 0:   t_range = slice(0, 4) # Hubs
            elif m_type == 1: t_range = slice(4, 6) # Obstacles
            elif m_type == 2: t_range = slice(6, 8) # Patrol zones
            elif m_type == 4: t_range = slice(8, 9) # Ghost Drone
            elif m_type == 5: t_range = slice(9, 10) # Recall Center
            else: continue 
            
            targets = self.tasks[t_range]
            if len(targets) == 0: continue
            
            diffs = targets - self.positions[i]
            dists = np.linalg.norm(diffs, axis=1)
            best_local_idx = np.argmin(dists)
            
            # Map back to global tasks index
            self.assigned_tasks[i] = t_range.start + best_local_idx
            self.bids[i] = dists[best_local_idx]

    def calculate_task_steer(self):
        """M3-Level Mission Logic: Dynamically driven by assigned_task indices"""
        ts = np.zeros_like(self.positions)
        alive = np.where(~self.dead_mask)[0]
        
        # Decrement mission timers
        self.mission_timer[alive] -= config.dt
        
        # 1. Handle "Finished" drones -> Seek Allocator
        done = (self.mission_timer <= 0) & (self.mission_type != 0)
        self.mission_type[done] = 0
        self.assigned_tasks[done] = -1 # Triggers auction for Hub (0-3)

        for i in alive:
            m_type = self.mission_type[i]
            tid = self.assigned_tasks[i]
            
            # Mission type 6 (COVERAGE_PATROL): Use task auction but with group cohesion
            if m_type == 6:
                if tid == -1: continue  # Wait for auction result
                
                # Will be handled below with added group cohesion boost
                m_type = 2  # Treat as patrol mission with cohesion enhancement
            
            if tid == -1: continue # Waiting for auction result

            pos = self.positions[i]
            target_pos = self.tasks[tid]
            
            if m_type == 0: # SEEKING ALLOCATOR HUB
                diff = target_pos - pos
                if np.linalg.norm(diff) < 35: # Reached Hub
                    # Handoff: Assign a new mission role (Circle, Patrol, or Intercept)
                    self.mission_type[i] = np.random.choice([1, 2, 4])
                    self.mission_timer[i] = np.random.uniform(15, 35)
                    self.assigned_tasks[i] = -1 # Triggers auction for Mission Target (4-8)
                    self.bids[i] = np.inf
                else:
                    ts[i] = self.steer(diff[np.newaxis, :], subtract_vels=self.velocities[i:i+1])[0]

            elif m_type == 1: # MISSION: CIRCLE POINT
                diff = pos - target_pos
                dist = np.linalg.norm(diff)
                radius = 110.0
                
                # Tangential velocity + Radial correction
                # Orbit in XZ plane
                tangent = np.array([-diff[2], 0, diff[0]]) 
                if np.linalg.norm(tangent) > 0: tangent /= np.linalg.norm(tangent)
                
                correction = (radius - dist) * (diff / max(dist, 1))
                desired = (tangent * config.max_speed) + (correction * 1.5)
                ts[i] = self.steer(desired[np.newaxis, :], subtract_vels=self.velocities[i:i+1])[0]

            elif m_type == 2: # MISSION: PATROL UNVISITED AREA (Hybrid Intelligent Search)
                # Find current grid index
                v_res = [self.env.width/self.grid_res, self.env.height/self.grid_res, self.env.depth/self.grid_res]
                gx = np.clip(int(pos[0] / v_res[0]), 0, self.grid_res-1)
                gy = np.clip(int(pos[1] / v_res[1]), 0, self.grid_res-1)
                gz = np.clip(int(pos[2] / v_res[2]), 0, self.grid_res-1)
                
                # If current target is visited or timer expires, find a new 'Cold' zone
                if self.visited_grid[gx, gy, gz] or self.mission_timer[i] < 0.8:
                    # Search locally first (wider radius for type 6 coverage drones)
                    search_radius = 8 if self.mission_type[i] == 6 else 4
                    rx = slice(max(0, gx-search_radius), min(self.grid_res, gx+search_radius))
                    ry = slice(max(0, gy-search_radius), min(self.grid_res, gy+search_radius))
                    rz = slice(max(0, gz-search_radius), min(self.grid_res, gz+search_radius))
                    
                    local_subgrid = self.visited_grid[rx, ry, rz]
                    local_unvisited = np.argwhere(~local_subgrid)
                    
                    if len(local_unvisited) > 0:
                        # Pick a random nearby unvisited voxel
                        choice = local_unvisited[np.random.randint(len(local_unvisited))]
                        target_v = [choice[0] + rx.start, choice[1] + ry.start, choice[2] + rz.start]
                    else:
                        # Global Leap: Find any unvisited voxel in the world
                        global_unvisited = np.argwhere(~self.visited_grid)
                        if len(global_unvisited) > 0:
                            target_v = global_unvisited[np.random.randint(len(global_unvisited))]
                        else:
                            # 100% coverage, return to hubs
                            target_v = [gx, gy, gz] 
                            
                    target_pos = np.array(target_v) * v_res + (np.array(v_res)/2)
                    self.tasks[6 + (i%2)] = target_pos
                    self.mission_timer[i] = np.random.uniform(5, 10) # Reset search timer
                
                diff = self.tasks[6 + (i%2)] - pos
                ts[i] = self.steer(diff[np.newaxis, :], subtract_vels=self.velocities[i:i+1])[0]

            elif m_type == 4: # MISSION: INTERCEPT GHOST
                # High-speed interception/tracking
                diff = target_pos - pos
                dist = np.linalg.norm(diff)
                safe_radius = 40.0
                if dist < safe_radius:
                    # Orbit or back off if too close
                    tangent = np.array([-diff[2], 0, diff[0]])
                    if np.linalg.norm(tangent) > 0: tangent /= np.linalg.norm(tangent)
                    desired = tangent * config.max_speed
                else:
                    desired = diff
                ts[i] = self.steer(desired[np.newaxis, :], subtract_vels=self.velocities[i:i+1])[0]
                
            elif m_type == 5: # MISSION: RECALL (Loose Orbiting Cloud)
                # Drones orbit at different altitudes and radii around center
                diff = target_pos - pos
                dist = np.linalg.norm(diff)
                
                # Tangential vector for orbiting
                tangent = np.array([-diff[2], 0, diff[0]])
                if np.linalg.norm(tangent) > 0: tangent /= np.linalg.norm(tangent)
                
                # Loose orbit with vertical oscillation
                v_osc = 100 * math.sin(self.frame_count * 0.02 + i)
                radius_tgt = 80.0 + (i % 20) * 5
                
                if dist > radius_tgt:
                    # Pull in
                    desired = (diff * 0.4) + (tangent * config.max_speed * 0.7)
                else:
                    # Maintain orbit
                    desired = (tangent * config.max_speed)
                    
                # Add individual vertical layer
                desired += np.array([0, (target_pos[1] + v_osc - pos[1]) * 0.3, 0])
                
                ts[i] = self.steer(desired[np.newaxis, :], subtract_vels=self.velocities[i:i+1])[0]
                
        return ts

    def calculate_formation_steer(self):
        """M2 B2.5: 3D V-Formation Control"""
        fs = np.zeros_like(self.positions)
        alive = ~self.dead_mask
        if not np.any(alive): return fs
        
        center = np.mean(self.positions[alive], axis=0)
        avg_vel = np.mean(self.velocities[alive], axis=0)
        spd = np.linalg.norm(avg_vel)
        if spd < 1e-3: return fs
        
        dir_vec = avg_vel / spd
        side_vec = np.cross(dir_vec, [0,1,0])
        
        row = np.arange(self.num_boids) % 10
        col = np.arange(self.num_boids) // 10
        
        targets = (center[np.newaxis, :] 
                  - dir_vec[np.newaxis, :] * (row[:, np.newaxis] * 60) 
                  + side_vec[np.newaxis, :] * (col[:, np.newaxis] * 50 - 250))
        
        diff = targets - self.positions
        # Pass FULL N-dim array so steer()'s self.velocities[valid] has matching shape
        full_steer = self.steer(diff, subtract_vels=True)
        fs[alive] = full_steer[alive]
            
        return fs

    def update(self):
        self.frame_count += 1
        self.env.step(config.dt)
        self.check_fault_injection()
        
        # M3: Update Coverage Grid from Brain
        self.last_grid[:] = self.visited_grid # Shadow copy
        alive_mask = ~self.dead_mask
        if np.any(alive_mask):
            pos = self.positions[alive_mask]
            gx = np.clip((pos[:, 0] / (self.env.width/self.grid_res)).astype(int), 0, self.grid_res-1)
            gy = np.clip((pos[:, 1] / (self.env.height/self.grid_res)).astype(int), 0, self.grid_res-1)
            gz = np.clip((pos[:, 2] / (self.env.depth/self.grid_res)).astype(int), 0, self.grid_res-1)
            self.visited_grid[gx, gy, gz] = True

        alive = ~self.dead_mask
        if not np.any(alive): return
        
        self.accelerations = np.zeros((self.num_boids, 3))
        
        pairs, alive_idx = self.find_neighbors()
        
        # M2 Collision Detection (C2.5)
        if len(pairs) == 2 and len(pairs[0]) > 0:
            ii, jj = pairs
            dist = np.linalg.norm(self.positions[ii] - self.positions[jj], axis=1)
            # Collision radius = 10 units (approx sphere diameter)
            collisions = np.sum(dist < 10)
            self.collision_count += collisions // 2 # Double counted in pairs
        # Cache pairs so visualizer can draw neighbor lines
        if len(pairs) == 2 and len(pairs[0]) > 0:
            self._last_pairs_i = np.asarray(pairs[0])
            self._last_pairs_j = np.asarray(pairs[1])
        else:
            self._last_pairs_i = np.array([], dtype=int)
            self._last_pairs_j = np.array([], dtype=int)

        # M2-M3 Logic Execution
        self.auction_tasks(pairs)
        task_s = self.calculate_task_steer()
        form_s = self.calculate_formation_steer()
        
        # Boids forces only for NON-FAILED drones
        active_mask = alive & ~self.failed_mask
        active_idx = np.where(active_mask)[0]
        
        sep_s, aln_s, coh_s = self.compute_forces(pairs, active_idx)
        obs_s = self.obstacle_avoidance()
        
        # M3: Drones must also avoid FALLING drones
        f_obs_s = np.zeros_like(sep_s)
        if np.any(self.failed_mask):
            failed_pos = self.positions[self.failed_mask]
            for i in active_idx:
                diff = self.positions[i] - failed_pos
                dist = np.linalg.norm(diff, axis=1)
                # Avoid if closer than 15 units
                danger = dist < 15.0
                if np.any(danger):
                    f_obs_s[i] = np.mean(diff[danger], axis=0) * 1.5

        wall_s = self.boundary_avoidance()
        
        # Apply forces to ACTIVE drones
        self.accelerations[active_idx] += sep_s[active_idx] * config.separation_weight
        self.accelerations[active_idx] += aln_s[active_idx] * config.alignment_weight
        self.accelerations[active_idx] += coh_s[active_idx] * config.cohesion_weight
        self.accelerations[active_idx] += obs_s[active_idx] * config.obstacle_weight
        self.accelerations[active_idx] += f_obs_s[active_idx] * 2.0 # Failure avoidance
        self.accelerations[active_idx] += wall_s[active_idx] * config.boundary_weight
        self.accelerations[active_idx] += task_s[active_idx] * config.task_weight
        self.accelerations[active_idx] += form_s[active_idx] * config.formation_weight
        
        # M3: FAILED drones only experience Gravity and Air Resistance
        failed_idx = np.where(alive & self.failed_mask)[0]
        self.accelerations[failed_idx] = [0, -18.0, 0] # Strong gravity
        self.velocities[failed_idx] *= 0.98 # Air resistance to tumble nicely
        
        # Real-time interactive pull (User Click) - only for active drones
        waypoint_s = np.zeros((self.num_boids, 3))
        if hasattr(self.env, 'target_waypoint') and self.env.target_waypoint is not None:
            waypoint_diff = np.array(self.env.target_waypoint) - self.positions
            waypoint_s = self.steer(waypoint_diff, subtract_vels=True)
            wp_weight = getattr(config, 'waypoint_weight', 2.5)
            self.accelerations[active_idx] += waypoint_s[active_idx] * wp_weight 

        # Store for Visualization (Task C2.4)
        self.last_sep = sep_s * config.separation_weight
        self.last_aln = aln_s * config.alignment_weight
        self.last_coh = coh_s * config.cohesion_weight
        wp_weight = getattr(config, 'waypoint_weight', 2.5)
        self.last_waypoint = waypoint_s * wp_weight        
        
        self.velocities[alive] += self.accelerations[alive]

        speeds = np.linalg.norm(self.velocities[alive], axis=1)
        over = speeds > config.max_speed
        if np.any(over):
            idx = np.where(alive)[0][over]
            self.velocities[idx] = (self.velocities[idx] / speeds[over, np.newaxis]) * config.max_speed
            
        self.positions[alive] += self.velocities[alive] * config.dt

    def steer(self, vectors, subtract_vels=None):
        mags = np.linalg.norm(vectors, axis=1)
        valid = mags > 0
        res = np.zeros_like(vectors)
        res[valid] = (vectors[valid] / np.maximum(mags[valid, np.newaxis], 1e-9)) * config.max_speed
        
        if subtract_vels is not None:
            if isinstance(subtract_vels, bool) and subtract_vels is True:
                # Use default velocities (assumes shapes match)
                res[valid] -= self.velocities[valid]
            else:
                # Use provided velocity array
                res[valid] -= subtract_vels[valid]
                
        fmags = np.linalg.norm(res, axis=1)
        over = fmags > config.max_force
        res[over] = (res[over] / np.maximum(fmags[over, np.newaxis], 1e-9)) * config.max_force
        return res

    def inject_faults(self, percentage=0.15):
        """Randomly marks drones as 'failed' (B3.x)"""
        num_fail = int(self.num_boids * percentage)
        potential = np.where(~self.dead_mask & ~self.failed_mask)[0]
        if len(potential) == 0: return
        
        to_fail = np.random.choice(potential, min(num_fail, len(potential)), replace=False)
        self.failed_mask[to_fail] = True
        # Clear their current task assignment so others can bid for it
        self.assigned_tasks[to_fail] = -1
        self.bids[to_fail] = np.inf

    def reset_faults(self):
        """Restores all failed drones to active duty"""
        self.failed_mask[:] = False

    def recall_fleet(self):
        """M3 Final: Recall all drones to Mission 5 (Orbiting Cloud) at center."""
        alive = np.where(~self.dead_mask & ~self.failed_mask)[0]
        self.mission_type[alive] = 5
        self.assigned_tasks[alive] = -1 # Triggers auction for Task 9 (Center)
        self.mission_timer[alive] = 999.0 
        # Ensure Task 9 is set to world center
        center = np.array([self.env.width/2, self.env.height/2, self.env.depth/2])
        self.tasks[9] = center
        print(f"[M3 SUCCESS] Mission recall initiated to {center}")

