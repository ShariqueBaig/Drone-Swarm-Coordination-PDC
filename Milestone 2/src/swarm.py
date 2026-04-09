import numpy as np
import pygame
import config

class SwarmManager:
    def __init__(self, env):
        self.env = env
        np.random.seed(env.seed)
        self.num_boids = env.num_drones
        self.ids = np.arange(self.num_boids)
        self.positions = np.random.rand(self.num_boids, 2) * [env.width, env.height]
        self.velocities = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))

        # --- Milestone 2 Setup ---
        # Decentralized Task Allocation & Consensus (B2.2, B2.3)
        # Create some random tasks/waypoints
        self.tasks = np.random.rand(10, 2) * [env.width * 0.8, env.height * 0.8] + [env.width * 0.1, env.height * 0.1]
        self.assigned_tasks = np.full(self.num_boids, -1, dtype=int)
        self.bids = np.full(self.num_boids, np.inf)

    def update(self, dt=None):
        if dt is None:
            dt = config.dt
            
        # Reset accelerations
        self.accelerations = np.zeros((self.num_boids, 2))
        
        # Distance matrix (N x N)
        # diff[i, j] = pos[i] - pos[j]
        # This can be memory intensive for very large N, but fine for N <= 1000
        # pos_i (N, 1, 2) - pos_j (1, N, 2) -> (N, N, 2)
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        
        # Avoid division by zero on diagonal (distance to self is 0)
        np.fill_diagonal(dist_matrix, np.inf)

        # Boolean mask for neighbors
        mask = dist_matrix < config.perception_radius
        separation_mask = dist_matrix < config.safety_distance
        
        # B2.4 Local communication mask
        comm_mask = dist_matrix < getattr(config, 'communication_radius', 100)
        
        # --- Milestone 2: Local Communication & Auction Consensus ---
        self.auction_tasks(comm_mask)
        
        # calculate task steer (B2.2)
        task_steer = self.calculate_task_steer()
        
        # calculate formation steer (B2.5)
        formation_steer = self.calculate_formation_steer()
        
        # --- Separation ---
        # Weight by distance: diff / dist
        # Avoid division by zero (already handled by diagonal inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        # Only consider neighbors within safety distance
        separation_vectors[~separation_mask] = 0
        separation = np.sum(separation_vectors, axis=1)
        
        # --- Alignment ---
        # Average velocity of neighbors
        # velocities (1, N, 2)
        neighbor_velocities = self.velocities[np.newaxis, :, :]
        # element-wise multiply with mask (N, N, 1)
        alignment_sum = np.sum(neighbor_velocities * mask[:, :, np.newaxis], axis=1)
        neighbor_counts = np.sum(mask, axis=1)[:, np.newaxis]
        
        # --- Cohesion ---
        # Average position of neighbors
        neighbor_positions = self.positions[np.newaxis, :, :]
        cohesion_sum = np.sum(neighbor_positions * mask[:, :, np.newaxis], axis=1)
        
        # --- Apply Rules ---
        
        # Normalize and steer Separation
        sep_steer = self.steer(separation, 0)
        
        # Normalize and steer Alignment
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vel = alignment_sum / neighbor_counts
        avg_vel[np.isnan(avg_vel)] = 0
        align_steer = self.steer(avg_vel, 0, subtract_velocity=True)
        
        # Normalize and steer Cohesion
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_pos = cohesion_sum / neighbor_counts
        avg_pos[np.isnan(avg_pos)] = 0
        
        # Vector to center of mass
        vec_to_com = avg_pos - self.positions
        # If no neighbors, vec_to_com is -self.positions, which is wrong, so zero it out
        vec_to_com[neighbor_counts.flatten() == 0] = 0
        
        cohesion_steer = self.steer(vec_to_com, 0, subtract_velocity=True)

        # --- Obstacle Avoidance ---
        obstacle_steer = np.zeros((self.num_boids, 2))
        if self.env.obstacles:
            # Suffiyan's environment format: list of (x, y, r)
            obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
            obs_radii = np.array([ob[2] for ob in self.env.obstacles])
            
            # diff (N, M, 2)
            obs_diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
            obs_dist = np.linalg.norm(obs_diff, axis=2) # (N, M)
            
            # Avoidance if within perception + obstacle radius
            obs_mask = obs_dist < (config.perception_radius + obs_radii[np.newaxis, :])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Normalize diff and scale by 1/distance
                obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (obs_dist[:, :, np.newaxis] / obs_radii[np.newaxis, :, np.newaxis])
            
            obs_avoid_vec[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid_vec, axis=1)
            obstacle_steer = self.steer(obstacle_steer, 0)

        # Weights
        self.accelerations += sep_steer * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight
        self.accelerations += task_steer * getattr(config, 'task_weight', 2.0)
        self.accelerations += formation_steer * getattr(config, 'formation_weight', 0.5)
        
        # Limit acceleration? (Not strictly necessary if we clip velocity, but helps stability)

        # Update Velocity
        self.velocities += self.accelerations
        
        # Limit Speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed
        
        # Update Position
        self.positions += self.velocities * dt
        
        # Boundary Handling (Using Environment's resolve logic)
        for i in range(self.num_boids):
            new_pos, new_vel = self.env.resolve_boundary(self.positions[i], self.velocities[i], dt)
            self.positions[i] = new_pos
            self.velocities[i] = new_vel

        # Hard Collision Resolution
        self.resolve_collisions()

    def resolve_collisions(self):
        if not self.env.obstacles:
            return

        obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
        obs_radii = np.array([ob[2] for ob in self.env.obstacles])
        
        # Check all boids against all obstacles
        diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2) # (N, M)
        
        # Find collisions
        # Collision radius = obstacle_radius + boid_size (e.g. 5)
        collision_radius = obs_radii[np.newaxis, :] + 5
        collisions = dist < collision_radius
        
        # If any collision exists
        if np.any(collisions):
            boid_indices, obs_indices = np.where(collisions)
            
            for boid_idx, obs_idx in zip(boid_indices, obs_indices):
                # Vector from obstacle center to boid
                vec = self.positions[boid_idx] - obs_centers[obs_idx]
                d = np.linalg.norm(vec)
                
                if d == 0: 
                    vec = np.random.randn(2)
                    d = np.linalg.norm(vec)
                
                normal = vec / d
                overlap = collision_radius[0, obs_idx] - d
                self.positions[boid_idx] += normal * overlap
                
                v = self.velocities[boid_idx]
                dot = np.dot(v, normal)
                if dot < 0:
                    self.velocities[boid_idx] = v - 2 * dot * normal

    def steer(self, vectors, target=None, subtract_velocity=False):
        # Normalize
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid = magnitudes > 0
        
        result = np.zeros_like(vectors)
        result[valid] = (vectors[valid] / magnitudes[valid, np.newaxis]) * config.max_speed
        
        if subtract_velocity:
            result[valid] -= self.velocities[valid]
            
        # Limit Force
        # Re-using magnitudes variable to store force magnitudes
        force_mags = np.linalg.norm(result, axis=1)
        limit_force = force_mags > config.max_force
        result[limit_force] = (result[limit_force] / force_mags[limit_force, np.newaxis]) * config.max_force
        
        return result

    def auction_tasks(self, comm_mask):
        """ B2.2, B2.3, B2.4: Local communication & Consensus """
        # 1. Generate bids for unassigned tasks based on proximity
        unassigned_boids = self.assigned_tasks == -1
        if np.any(unassigned_boids) and len(self.tasks) > 0:
            diff_to_tasks = self.positions[unassigned_boids, np.newaxis, :] - self.tasks[np.newaxis, :, :]
            dist_to_tasks = np.linalg.norm(diff_to_tasks, axis=2)
            
            closest_tasks = np.argmin(dist_to_tasks, axis=1)
            min_dists = np.min(dist_to_tasks, axis=1)
            
            self.assigned_tasks[unassigned_boids] = closest_tasks
            self.bids[unassigned_boids] = min_dists
            
        # 2. Local Communication (B2.4) mapping 
        # Broadcast intents and resolve conflicts via Consensus (B2.3)
        for i in range(self.num_boids):
            task_i = self.assigned_tasks[i]
            if task_i == -1: continue
            
            neighbors = np.where(comm_mask[i])[0]
            for j in neighbors:
                if i != j and self.assigned_tasks[j] == task_i:
                    # Consensus Rule: Lower distance (bid) wins. Tie-breaker is lower ID.
                    if self.bids[j] < self.bids[i] or (self.bids[j] == self.bids[i] and self.ids[j] < self.ids[i]):
                        # i loses to j, drops the task
                        self.assigned_tasks[i] = -1
                        self.bids[i] = np.inf
                        break
                        
    def calculate_task_steer(self):
        """ Calculate steering force toward assigned tasks """
        task_steer = np.zeros((self.num_boids, 2))
        valid = self.assigned_tasks != -1
        if not np.any(valid):
            return task_steer
            
        target_positions = self.tasks[self.assigned_tasks[valid]]
        vec_to_target = np.zeros((self.num_boids, 2))
        vec_to_target[valid] = target_positions - self.positions[valid]
        
        # Check task completion
        distances = np.linalg.norm(vec_to_target, axis=1)
        task_radius = getattr(config, 'task_radius', 20)
        completed = (distances < task_radius) & valid
        if np.any(completed):
            self.assigned_tasks[completed] = -1
            self.bids[completed] = np.inf
            
        # Standard steer toward task
        task_steer = self.steer(vec_to_target, subtract_velocity=True)
        return task_steer
        
    def calculate_formation_steer(self):
        """ B2.5 Formation control (V-shape) via relative centroid steering """
        steer = np.zeros((self.num_boids, 2))
        centroid = np.mean(self.positions, axis=0)
        vel_centroid = np.mean(self.velocities, axis=0)
        
        speed = np.linalg.norm(vel_centroid)
        # Avoid zero division
        if speed < 1e-3: 
            return steer
            
        dir_vec = vel_centroid / speed
        perp_vec = np.array([-dir_vec[1], dir_vec[0]])
        
        for i in range(self.num_boids):
            row = i // 2
            side = 1 if i % 2 == 0 else -1
            if i == 0: row, side = 0, 0
            
            # V formation offsets
            offset = -dir_vec * (row * 30) + perp_vec * (side * row * 30)
            target = centroid + offset
            
            vec = target - self.positions[i]
            steer[i] = vec
            
        return self.steer(steer, subtract_velocity=True)
