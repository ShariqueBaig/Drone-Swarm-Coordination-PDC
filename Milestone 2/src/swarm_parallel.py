"""
swarm_parallel.py - Parallel SwarmManager with multiprocessing
Milestone 2 - Ashhal Aamir (Optimization & Parallelism)
"""

import numpy as np
import multiprocessing as mp
from functools import partial
import config
from environment import Environment

class SwarmManagerParallel:
    def __init__(self, env, num_processes=None):
        self.env = env
        self.num_boids = config.num_boids
        self.num_processes = num_processes or min(mp.cpu_count(), 4)
        
        # Limit processes to number of drones
        if self.num_processes > self.num_boids:
            self.num_processes = self.num_boids
        
        np.random.seed(env.seed)
        self.ids = np.arange(self.num_boids)
        self.positions = np.random.rand(self.num_boids, 2) * [env.width, env.height]
        self.velocities = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))

        # Milestone 2 Setup
        self.tasks = np.random.rand(10, 2) * [env.width * 0.8, env.height * 0.8] + [env.width * 0.1, env.height * 0.1]
        self.assigned_tasks = np.full(self.num_boids, -1, dtype=int)
        self.bids = np.full(self.num_boids, np.inf)
        
        self.frame_count = 0
        
        print(f"🚀 Parallel SwarmManager initialized with {self.num_processes} CPU cores")

    def update(self, dt=None):
        """Main update - uses sequential for now (parallel coming)"""
        if dt is None:
            dt = config.dt
            
        self.frame_count += 1
        self.accelerations = np.zeros((self.num_boids, 2))
        
        # Distance matrix (shared calculation)
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        mask = dist_matrix < config.perception_radius
        separation_mask = dist_matrix < config.safety_distance
        comm_mask = dist_matrix < getattr(config, 'communication_radius', 100)
        
        # Auction tasks (sequential for now)
        self._auction_tasks(comm_mask)
        
        # Task steer
        task_steer = self._calculate_task_steer()
        
        # Formation steer
        formation_steer = self._calculate_formation_steer()
        
        # Separation, Alignment, Cohesion (vectorized)
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        separation_vectors[~separation_mask] = 0
        separation = np.sum(separation_vectors, axis=1)
        
        neighbor_velocities = self.velocities[np.newaxis, :, :]
        alignment_sum = np.sum(neighbor_velocities * mask[:, :, np.newaxis], axis=1)
        neighbor_counts = np.sum(mask, axis=1)[:, np.newaxis]
        neighbor_counts = np.maximum(neighbor_counts, 1)
        
        neighbor_positions = self.positions[np.newaxis, :, :]
        cohesion_sum = np.sum(neighbor_positions * mask[:, :, np.newaxis], axis=1)
        
        sep_steer = self._steer(separation)
        
        avg_vel = alignment_sum / neighbor_counts
        avg_vel[np.isnan(avg_vel)] = 0
        align_steer = self._steer(avg_vel, subtract_velocity=True)
        
        avg_pos = cohesion_sum / neighbor_counts
        avg_pos[np.isnan(avg_pos)] = 0
        vec_to_com = avg_pos - self.positions
        vec_to_com[neighbor_counts.flatten() == 0] = 0
        cohesion_steer = self._steer(vec_to_com, subtract_velocity=True)
        
        # Obstacle avoidance
        obstacle_steer = self._calculate_obstacle_steer()
        
        # Apply weights
        self.accelerations += sep_steer * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight
        self.accelerations += task_steer * getattr(config, 'task_weight', 2.0)
        self.accelerations += formation_steer * getattr(config, 'formation_weight', 0.5)
        
        # Update Velocity
        self.velocities += self.accelerations
        
        # Limit Speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        if np.any(limit_mask):
            speeds_safe = np.maximum(speeds[limit_mask], 1e-6)
            self.velocities[limit_mask] = (self.velocities[limit_mask] / speeds_safe[:, np.newaxis]) * config.max_speed
        
        # Update Position
        self.positions += self.velocities * dt
        
        # Boundary Handling
        self._resolve_boundary(dt)
        
        # Collision Resolution
        self._resolve_collisions()

    def _auction_tasks(self, comm_mask):
        """Auction tasks with conflict resolution"""
        unassigned_boids = self.assigned_tasks == -1
        if np.any(unassigned_boids) and len(self.tasks) > 0:
            diff_to_tasks = self.positions[unassigned_boids, np.newaxis, :] - self.tasks[np.newaxis, :, :]
            dist_to_tasks = np.linalg.norm(diff_to_tasks, axis=2)
            
            closest_tasks = np.argmin(dist_to_tasks, axis=1)
            min_dists = np.min(dist_to_tasks, axis=1)
            
            self.assigned_tasks[unassigned_boids] = closest_tasks
            self.bids[unassigned_boids] = min_dists
        
        # Conflict resolution (sequential but can be parallelized)
        for i in range(self.num_boids):
            if self.assigned_tasks[i] == -1:
                continue
            neighbors = np.where(comm_mask[i])[0]
            for j in neighbors:
                if i != j and self.assigned_tasks[j] == self.assigned_tasks[i]:
                    if self.bids[j] < self.bids[i] or (self.bids[j] == self.bids[i] and self.ids[j] < self.ids[i]):
                        self.assigned_tasks[i] = -1
                        self.bids[i] = np.inf
                        break

    def _calculate_task_steer(self):
        """Calculate steering force toward assigned tasks"""
        task_steer = np.zeros((self.num_boids, 2))
        valid = self.assigned_tasks != -1
        if not np.any(valid):
            return task_steer
            
        target_positions = self.tasks[self.assigned_tasks[valid]]
        vec_to_target = np.zeros((self.num_boids, 2))
        vec_to_target[valid] = target_positions - self.positions[valid]
        
        distances = np.linalg.norm(vec_to_target, axis=1)
        task_radius = getattr(config, 'task_radius', 20)
        completed = (distances < task_radius) & valid
        if np.any(completed):
            self.assigned_tasks[completed] = -1
            self.bids[completed] = np.inf
            
        task_steer = self._steer(vec_to_target, subtract_velocity=True)
        return task_steer

    def _calculate_formation_steer(self):
        """Formation control (vectorized)"""
        centroid = np.mean(self.positions, axis=0)
        vel_centroid = np.mean(self.velocities, axis=0)
        speed = np.linalg.norm(vel_centroid)
        
        if speed < 1e-3:
            return np.zeros((self.num_boids, 2))
        
        speed_safe = max(speed, 1e-6)
        dir_vec = vel_centroid / speed_safe
        perp_vec = np.array([-dir_vec[1], dir_vec[0]])
        
        rows = np.arange(self.num_boids) // 2
        sides = np.where(np.arange(self.num_boids) % 2 == 0, 1, -1)
        rows[0] = 0
        sides[0] = 0
        
        row_factor = rows * 30
        offsets = (-dir_vec * row_factor[:, np.newaxis] + 
                   perp_vec * (sides * row_factor)[:, np.newaxis])
        
        targets = centroid + offsets
        steer = targets - self.positions
        
        return self._steer(steer, subtract_velocity=True)

    def _calculate_obstacle_steer(self):
        """Obstacle avoidance"""
        obstacle_steer = np.zeros((self.num_boids, 2))
        if not self.env.obstacles:
            return obstacle_steer
            
        obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
        obs_radii = np.array([ob[2] for ob in self.env.obstacles])
        
        obs_diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
        obs_dist = np.linalg.norm(obs_diff, axis=2)
        obs_mask = obs_dist < (config.perception_radius + obs_radii[np.newaxis, :])
        
        with np.errstate(divide='ignore', invalid='ignore'):
            obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (obs_dist[:, :, np.newaxis] / obs_radii[np.newaxis, :, np.newaxis])
        
        obs_avoid_vec[~obs_mask] = 0
        obstacle_steer = np.sum(obs_avoid_vec, axis=1)
        obstacle_steer = self._steer(obstacle_steer)
        
        return obstacle_steer

    def _resolve_boundary(self, dt):
        """Vectorized boundary handling"""
        px = self.positions[:, 0].copy()
        py = self.positions[:, 1].copy()
        vx = self.velocities[:, 0].copy()
        vy = self.velocities[:, 1].copy()

        if self.env.boundary == "wrap":
            px = px % self.env.width
            py = py % self.env.height
        else:
            left_mask = px < 0
            right_mask = px > self.env.width
            bottom_mask = py < 0
            top_mask = py > self.env.height
            
            vx[left_mask] = -vx[left_mask]
            vx[right_mask] = -vx[right_mask]
            vy[bottom_mask] = -vy[bottom_mask]
            vy[top_mask] = -vy[top_mask]
            
            px[left_mask] = 0
            px[right_mask] = self.env.width
            py[bottom_mask] = 0
            py[top_mask] = self.env.height
        
        margin = self.env.boundary_margin
        k = self.env.boundary_repulsion_strength
        
        repulse_x = np.zeros(self.num_boids)
        repulse_y = np.zeros(self.num_boids)
        
        near_left = px < margin
        near_right = px > self.env.width - margin
        near_bottom = py < margin
        near_top = py > self.env.height - margin
        
        eps = 1e-6
        
        if np.any(near_left):
            repulse_x[near_left] += k / (px[near_left] ** 2 + eps)
        if np.any(near_right):
            repulse_x[near_right] -= k / ((self.env.width - px[near_right]) ** 2 + eps)
        if np.any(near_bottom):
            repulse_y[near_bottom] += k / (py[near_bottom] ** 2 + eps)
        if np.any(near_top):
            repulse_y[near_top] -= k / ((self.env.height - py[near_top]) ** 2 + eps)
        
        vx += repulse_x * dt
        vy += repulse_y * dt
        
        self.positions[:, 0] = px
        self.positions[:, 1] = py
        self.velocities[:, 0] = vx
        self.velocities[:, 1] = vy

    def _resolve_collisions(self):
        """Collision resolution"""
        if not self.env.obstacles:
            return

        obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
        obs_radii = np.array([ob[2] for ob in self.env.obstacles])
        
        diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)
        
        collision_radius = obs_radii[np.newaxis, :] + 5
        collisions = dist < collision_radius
        
        if np.any(collisions):
            boid_indices, obs_indices = np.where(collisions)
            
            for boid_idx, obs_idx in zip(boid_indices, obs_indices):
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

    def _steer(self, vectors, subtract_velocity=False):
        """Steering helper"""
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid = magnitudes > 1e-6
        
        result = np.zeros_like(vectors)
        if np.any(valid):
            mags_safe = np.maximum(magnitudes[valid], 1e-6)
            result[valid] = (vectors[valid] / mags_safe[:, np.newaxis]) * config.max_speed
        
        if subtract_velocity and np.any(valid):
            result[valid] -= self.velocities[valid]
            
        force_mags = np.linalg.norm(result, axis=1)
        limit_force = force_mags > config.max_force
        if np.any(limit_force):
            force_mags_safe = np.maximum(force_mags[limit_force], 1e-6)
            result[limit_force] = (result[limit_force] / force_mags_safe[:, np.newaxis]) * config.max_force
        
        return result