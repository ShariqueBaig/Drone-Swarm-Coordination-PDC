import numpy as np
import pygame
import config

class Flock:
    def __init__(self):
        self.num_boids = config.num_boids
        self.positions = np.random.rand(self.num_boids, 2) * [config.width, config.height]
        self.velocities = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))
        self.obstacles = []

    def update(self):
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
        
        # --- Separation ---
        # Weight by distance: diff / dist
        # Avoid division by zero (already handled by diagonal inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        # Only consider neighbors
        separation_vectors[~mask] = 0
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
        if self.obstacles:
            obs_array = np.array(self.obstacles) # (M, 2)
            # diff (N, M, 2)
            obs_diff = self.positions[:, np.newaxis, :] - obs_array[np.newaxis, :, :]
            obs_dist = np.linalg.norm(obs_diff, axis=2) # (N, M)
            
            obs_mask = obs_dist < (config.perception_radius + config.obstacle_radius)
            
            with np.errstate(divide='ignore', invalid='ignore'):
                # Stronger avoidance (inverse distance squared or similar)
                # Normalizing diff gives direction. Dividing by dist gives 1/d. 
                # Dividing by dist^2 gives 1/d^2 force.
                obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (obs_dist[:, :, np.newaxis] / config.obstacle_radius)
            
            obs_avoid_vec[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid_vec, axis=1)
            obstacle_steer = self.steer(obstacle_steer, 0)

        # Weights
        self.accelerations += sep_steer * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight
        
        # Limit acceleration? (Not strictly necessary if we clip velocity, but helps stability)

        # Update Velocity
        self.velocities += self.accelerations
        
        # Limit Speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed
        
        # Update Position
        self.positions += self.velocities
        
        # Wrap Edges
        self.positions[:, 0] = np.mod(self.positions[:, 0], config.width)
        self.positions[:, 1] = np.mod(self.positions[:, 1], config.height)

        # Hard Collision Resolution
        self.resolve_collisions()

    def resolve_collisions(self):
        if not self.obstacles:
            return

        obs_array = np.array(self.obstacles)
        # Check all boids against all obstacles
        # pos (N, 1, 2) - obs (1, M, 2)
        diff = self.positions[:, np.newaxis, :] - obs_array[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2) # (N, M)
        
        # Find collisions
        # Collision radius = obstacle_radius + safety_margin (e.g. 5 for boid size)
        collision_radius = config.obstacle_radius + 5
        collisions = dist < collision_radius
        
        # If any collision exists
        if np.any(collisions):
            # For each boid, find the closest obstacle it collided with (if any)
            # This is a bit simplified, handling multiple simultaneous collisions is hard
            # We just take the first one or iterate.
            
            # Get indices of (boid_idx, obs_idx) having collision
            boid_indices, obs_indices = np.where(collisions)
            
            for boid_idx, obs_idx in zip(boid_indices, obs_indices):
                # Vector from obstacle to boid
                vec = self.positions[boid_idx] - self.obstacles[obs_idx]
                d = np.linalg.norm(vec)
                
                if d == 0: # Exact overlap, push random
                    vec = np.random.randn(2)
                    d = np.linalg.norm(vec)
                
                # Normalize
                normal = vec / d
                
                # Push out
                overlap = collision_radius - d
                self.positions[boid_idx] += normal * overlap
                
                # Reflect velocity: v' = v - 2 * (v . n) * n
                v = self.velocities[boid_idx]
                dot = np.dot(v, normal)
                
                # Only reflect if moving towards the obstacle
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

    def draw(self, screen):
        # Draw all boids
        # Calculate angles
        angles = np.arctan2(self.velocities[:, 1], self.velocities[:, 0])
        
        # Triangle points
        r = 6
        # Cos/Sin for all boids
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        # Tip
        p1_x = self.positions[:, 0] + cos_a * r * 2
        p1_y = self.positions[:, 1] + sin_a * r * 2
        
        # Back Left (angle + 2.5 rad)
        cos_a_l = np.cos(angles + 2.5)
        sin_a_l = np.sin(angles + 2.5)
        p2_x = self.positions[:, 0] + cos_a_l * r
        p2_y = self.positions[:, 1] + sin_a_l * r
        
        # Back Right (angle - 2.5 rad)
        cos_a_r = np.cos(angles - 2.5)
        sin_a_r = np.sin(angles - 2.5)
        p3_x = self.positions[:, 0] + cos_a_r * r
        p3_y = self.positions[:, 1] + sin_a_r * r
        
        # Stack into vertices (N, 3, 2)
        p1 = np.stack((p1_x, p1_y), axis=1)
        p2 = np.stack((p2_x, p2_y), axis=1)
        p3 = np.stack((p3_x, p3_y), axis=1)
        
        # Calculate line end points (start is p1 - tip, end is p1 + velocity_direction * 20)
        line_length = 20
        line_end_x = p1_x + cos_a * line_length
        line_end_y = p1_y + sin_a * line_length
        line_end = np.stack((line_end_x, line_end_y), axis=1)

        # Create a transparent surface for lines
        item_surf = pygame.Surface((config.width, config.height), pygame.SRCALPHA)
        
        # Ideally we would use a batch draw, but Pygame doesn't support polygon batch well.
        # We simulate batch drawing loops, but since N is small (< 1000) this loop is faster than O(N^2) logic.
        for i in range(self.num_boids):
             pygame.draw.polygon(screen, config.BOID_COLOR, [p1[i], p2[i], p3[i]])
             # Draw direction line on transparent surface
             pygame.draw.line(item_surf, (255, 255, 255, 50), p1[i], line_end[i], 1)
        
        # Blit the transparent surface
        screen.blit(item_surf, (0, 0))

        for obstacle in self.obstacles:
            pygame.draw.circle(screen, config.OBSTACLE_COLOR, (int(obstacle[0]), int(obstacle[1])), config.obstacle_radius)

    def add_obstacle(self, x, y):
        self.obstacles.append((x, y))

    def remove_obstacle(self, x, y):
        to_remove = []
        for obstacle in self.obstacles:
            dist = np.linalg.norm(np.array(obstacle) - np.array([x, y]))
            if dist < config.obstacle_radius:
                to_remove.append(obstacle)
        
        for item in to_remove:
            self.obstacles.remove(item)
