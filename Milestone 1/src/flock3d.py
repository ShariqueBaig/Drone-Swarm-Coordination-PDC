import numpy as np
import config

class Flock3D:
    def __init__(self):
        self.num_boids = config.num_boids
        # 3D Positions: (x, y, z)
        # We use a cubic bounds defined by width/height/depth or similar
        # For simplicity, we can use config.width for x/z and config.height for y
        self.bounds = np.array([config.width, config.height, config.width]) 
        
        self.positions = np.random.rand(self.num_boids, 3) * self.bounds
        self.velocities = (np.random.rand(self.num_boids, 3) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 3))
        self.obstacles = []

    def update(self):
        # Reset accelerations
        self.accelerations = np.zeros((self.num_boids, 3))
        
        # Distance matrix (N x N)
        # diff[i, j] = pos[i] - pos[j]
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        
        # Avoid division by zero
        np.fill_diagonal(dist_matrix, np.inf)

        # Boolean mask for neighbors
        mask = dist_matrix < config.perception_radius
        
        # --- Separation ---
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        separation_vectors[~mask] = 0
        separation = np.sum(separation_vectors, axis=1)
        
        # --- Alignment ---
        neighbor_velocities = self.velocities[np.newaxis, :, :]
        alignment_sum = np.sum(neighbor_velocities * mask[:, :, np.newaxis], axis=1)
        neighbor_counts = np.sum(mask, axis=1)[:, np.newaxis]
        
        # --- Cohesion ---
        neighbor_positions = self.positions[np.newaxis, :, :]
        cohesion_sum = np.sum(neighbor_positions * mask[:, :, np.newaxis], axis=1)
        
        # --- Apply Rules ---
        sep_steer = self.steer(separation)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vel = alignment_sum / neighbor_counts
        avg_vel[np.isnan(avg_vel)] = 0
        align_steer = self.steer(avg_vel, subtract_velocity=True)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            avg_pos = cohesion_sum / neighbor_counts
        avg_pos[np.isnan(avg_pos)] = 0
        
        vec_to_com = avg_pos - self.positions
        vec_to_com[neighbor_counts.flatten() == 0] = 0
        cohesion_steer = self.steer(vec_to_com, subtract_velocity=True)

        # --- Obstacle Avoidance (3D) ---
        obstacle_steer = np.zeros((self.num_boids, 3))
        if self.obstacles:
            obs_array = np.array(self.obstacles) # (M, 3)
            # diff (N, M, 3)
            obs_diff = self.positions[:, np.newaxis, :] - obs_array[np.newaxis, :, :]
            obs_dist = np.linalg.norm(obs_diff, axis=2) # (N, M)
            
            # Use same radius/logic as 2D for now
            obs_mask = obs_dist < (config.perception_radius + config.obstacle_radius)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                 # Stronger avoidance (inverse distance squared)
                obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (obs_dist[:, :, np.newaxis] / config.obstacle_radius)
            
            obs_avoid_vec[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid_vec, axis=1)
            obstacle_steer = self.steer(obstacle_steer)

        # --- Boundary Avoidance (3D) ---
        boundary_steer = np.zeros((self.num_boids, 3))
        margin = config.boundary_margin
        
        # Avoid Min-X, Max-X
        # If pos < margin, steer +X. If pos > bounds - margin, steer -X
        boundary_steer[self.positions[:, 0] < margin, 0] = config.max_speed
        boundary_steer[self.positions[:, 0] > self.bounds[0] - margin, 0] = -config.max_speed
        
        # Avoid Min-Y, Max-Y
        boundary_steer[self.positions[:, 1] < margin, 1] = config.max_speed
        boundary_steer[self.positions[:, 1] > self.bounds[1] - margin, 1] = -config.max_speed
        
        # Avoid Min-Z, Max-Z
        boundary_steer[self.positions[:, 2] < margin, 2] = config.max_speed
        boundary_steer[self.positions[:, 2] > self.bounds[2] - margin, 2] = -config.max_speed

        boundary_steer = self.steer(boundary_steer)

        # Weights
        self.accelerations += sep_steer * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight
        self.accelerations += boundary_steer * config.boundary_weight
        
        # Update Velocity
        self.velocities += self.accelerations
        
        # Limit Speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed
        
        # Update Position
        self.positions += self.velocities
        
        # Hard Constraints (Clamp to Bounds)
        # Instead of wrapping, we keep them inside the box
        self.positions = np.clip(self.positions, 0, self.bounds)

        # Hard Collision (Simple push out)
        # self.resolve_collisions() # Need to adapt for 3D obstacles

    def steer(self, vectors, subtract_velocity=False):
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid = magnitudes > 0
        
        result = np.zeros_like(vectors)
        result[valid] = (vectors[valid] / magnitudes[valid, np.newaxis]) * config.max_speed
        
        if subtract_velocity:
            result[valid] -= self.velocities[valid]
            
        force_mags = np.linalg.norm(result, axis=1)
        limit_force = force_mags > config.max_force
        result[limit_force] = (result[limit_force] / force_mags[limit_force, np.newaxis]) * config.max_force
        
        return result

    def add_obstacle(self, position):
        # position tuple (x, y, z)
        self.obstacles.append(position)
