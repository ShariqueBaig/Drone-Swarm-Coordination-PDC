import numpy as np
import config
from spatial_grid import SpatialGrid
from quadtree import QuadTree
from performance_logger import PerformanceLogger

class SwarmManagerOptimized:
    def __init__(self, env):
        self.env = env
        np.random.seed(config.seed)
        self.num_boids = config.num_boids
        self.ids = np.arange(self.num_boids)
        self.positions = np.random.rand(self.num_boids, 2) * [env.width, env.height]
        self.velocities = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))
        
        # Initialize spatial grid
        self.spatial_grid = SpatialGrid(
            cell_size=config.perception_radius,
            width=env.width,
            height=env.height
        )
        
        # Performance tracking
        self.frame_count = 0
        self.logger = PerformanceLogger("optimized_benchmark.csv")
        
        # For benchmarking different methods
        self.use_method = 'grid'  # Options: 'naive', 'grid', 'quadtree'
        
        # Statistics
        self.avg_neighbors = 0
        self.neighbor_counts = np.zeros(self.num_boids)

    def find_neighbors_naive(self):
        """Original O(n²) neighbor detection (fixed)"""
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Create neighbor mask
        neighbor_mask = dist_matrix < config.perception_radius
        
        # Update neighbor counts and avg_neighbors
        self.neighbor_counts = np.sum(neighbor_mask, axis=1)
        self.avg_neighbors = np.mean(self.neighbor_counts)
        
        return neighbor_mask

    def find_neighbors_grid(self):
        """Optimized O(n) neighbor detection using spatial grid"""
        # Clear and rebuild grid
        self.spatial_grid.clear()
        self.spatial_grid.insert_drones(self.positions, self.ids)
        
        # Build neighbor mask
        neighbor_mask = np.zeros((self.num_boids, self.num_boids), dtype=bool)
        total_neighbors = 0
        self.neighbor_counts = np.zeros(self.num_boids)
        
        for i, pos in enumerate(self.positions):
            neighbors = self.spatial_grid.get_neighbors(pos, config.perception_radius)
            count = 0
            for idx, _ in neighbors:
                if idx != i:
                    neighbor_mask[i, idx] = True
                    count += 1
                    total_neighbors += 1
            self.neighbor_counts[i] = count
        
        self.avg_neighbors = total_neighbors / self.num_boids if self.num_boids > 0 else 0
        return neighbor_mask

    def find_neighbors_quadtree(self):
        """Optimized neighbor detection using quadtree"""
        # Rebuild quadtree
        quadtree = QuadTree((0, 0, self.env.width, self.env.height))
        
        for i, pos in enumerate(self.positions):
            quadtree.insert(i, pos)
        
        # Build neighbor mask
        neighbor_mask = np.zeros((self.num_boids, self.num_boids), dtype=bool)
        total_neighbors = 0
        self.neighbor_counts = np.zeros(self.num_boids)
        
        for i, pos in enumerate(self.positions):
            neighbors = quadtree.query_radius(pos, config.perception_radius)
            count = 0
            for idx, _ in neighbors:
                if idx != i:
                    neighbor_mask[i, idx] = True
                    count += 1
                    total_neighbors += 1
            self.neighbor_counts[i] = count
        
        self.avg_neighbors = total_neighbors / self.num_boids if self.num_boids > 0 else 0
        return neighbor_mask

    def compute_forces_from_mask(self, mask):
        """Compute flocking forces using neighbor mask"""
        # Calculate distance matrix and separation mask
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        
        separation_mask = dist_matrix < config.safety_distance
        
        # --- Separation ---
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        separation_vectors[~separation_mask] = 0
        separation = np.sum(separation_vectors, axis=1)
        
        # --- Alignment ---
        neighbor_velocities = self.velocities[np.newaxis, :, :]
        alignment_sum = np.sum(neighbor_velocities * mask[:, :, np.newaxis], axis=1)
        neighbor_counts = np.sum(mask, axis=1)[:, np.newaxis]
        neighbor_counts = np.maximum(neighbor_counts, 1)  # Avoid division by zero
        
        # --- Cohesion ---
        neighbor_positions = self.positions[np.newaxis, :, :]
        cohesion_sum = np.sum(neighbor_positions * mask[:, :, np.newaxis], axis=1)
        
        # --- Apply Rules ---
        sep_steer = self.steer(separation)
        
        avg_vel = alignment_sum / neighbor_counts
        align_steer = self.steer(avg_vel, subtract_velocity=True)
        
        avg_pos = cohesion_sum / neighbor_counts
        vec_to_com = avg_pos - self.positions
        cohesion_steer = self.steer(vec_to_com, subtract_velocity=True)
        
        return sep_steer, align_steer, cohesion_steer

    def update(self, dt=None):
        if dt is None:
            dt = config.dt
        
        self.frame_count += 1
        self.logger.start_frame()
            
        # Reset accelerations
        self.accelerations = np.zeros((self.num_boids, 2))
        
        # Find neighbors using selected method
        if self.use_method == 'naive':
            mask = self.find_neighbors_naive()
        elif self.use_method == 'grid':
            mask = self.find_neighbors_grid()
        else:  # quadtree
            mask = self.find_neighbors_quadtree()
        
        # Compute forces using the mask
        sep_steer, align_steer, cohesion_steer = self.compute_forces_from_mask(mask)

        # --- Obstacle Avoidance ---
        obstacle_steer = np.zeros((self.num_boids, 2))
        if self.env.obstacles:
            obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
            obs_radii = np.array([ob[2] for ob in self.env.obstacles])
            
            obs_diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
            obs_dist = np.linalg.norm(obs_diff, axis=2)
            
            obs_mask = obs_dist < (config.perception_radius + obs_radii[np.newaxis, :])
            
            with np.errstate(divide='ignore', invalid='ignore'):
                obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (obs_dist[:, :, np.newaxis] / obs_radii[np.newaxis, :, np.newaxis])
            
            obs_avoid_vec[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid_vec, axis=1)
            obstacle_steer = self.steer(obstacle_steer)

        # Apply weights
        self.accelerations += sep_steer * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight

        # Update Velocity
        self.velocities += self.accelerations
        
        # Limit Speed
        speeds = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed
        
        # Update Position
        self.positions += self.velocities * dt
        
        # Boundary Handling
        for i in range(self.num_boids):
            new_pos, new_vel = self.env.resolve_boundary(self.positions[i], self.velocities[i], dt)
            self.positions[i] = new_pos
            self.velocities[i] = new_vel

        # Hard Collision Resolution
        self.resolve_collisions()
        
        # Log performance data
        self.logger.end_frame(self)

    def resolve_collisions(self):
        """Same as original resolve_collisions"""
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

    def steer(self, vectors, subtract_velocity=False):
        """Same as original steer method"""
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

    def set_method(self, method):
        """Switch between optimization methods"""
        if method in ['naive', 'grid', 'quadtree']:
            self.use_method = method
            print(f"Switched to {method} method")
        else:
            print(f"Invalid method: {method}. Use 'naive', 'grid', or 'quadtree'")