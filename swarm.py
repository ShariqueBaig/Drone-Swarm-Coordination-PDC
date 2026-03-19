import numpy as np
import random

# default behavior weights (can be tuned per mission)
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0

class Drone:
    def __init__(self, x, y, max_speed=5.0):
        self.position = np.array([float(x), float(y)], dtype=float)
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)], dtype=float)
        self.max_speed = float(max_speed)
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * self.max_speed

    def separation(self, neighbors):
        steer = np.zeros(2, dtype=float)
        if len(neighbors) == 0:
            return steer

        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                steer += diff / (dist + 1e-5)

        if np.linalg.norm(steer) > 0:
            steer = steer / np.linalg.norm(steer) * self.max_speed
        return steer

    def alignment(self, neighbors):
        if len(neighbors) == 0:
            return np.zeros(2, dtype=float)

        avg_vel = np.mean([neighbor.velocity for neighbor in neighbors], axis=0)
        norm = np.linalg.norm(avg_vel)
        if norm > 0:
            return avg_vel / norm * self.max_speed
        return np.zeros(2, dtype=float)

    def cohesion(self, neighbors):
        if len(neighbors) == 0:
            return np.zeros(2, dtype=float)

        center = np.mean([neighbor.position for neighbor in neighbors], axis=0)
        steer = center - self.position
        norm = np.linalg.norm(steer)
        if norm > 0:
            return steer / norm * self.max_speed
        return np.zeros(2, dtype=float)

    def update(self, neighbors, delta_t=0.1, obstacles=None, safety_distance=10.0):
        sep = self.separation(neighbors) * SEPARATION_WEIGHT
        ali = self.alignment(neighbors) * ALIGNMENT_WEIGHT
        coh = self.cohesion(neighbors) * COHESION_WEIGHT

        force = sep + ali + coh

        if obstacles is not None:
            obs_force = np.zeros(2, dtype=float)
            for ox, oy, oradius in obstacles:
                obs_vec = self.position - np.array([ox, oy])
                dist = np.linalg.norm(obs_vec)
                if dist < oradius + safety_distance and dist > 0:
                    obs_force += obs_vec / (dist + 1e-5)
            if np.linalg.norm(obs_force) > 0:
                obs_force = obs_force / np.linalg.norm(obs_force) * self.max_speed * 2
            force += obs_force

        if np.linalg.norm(force) > 0:
            self.velocity = force / np.linalg.norm(force) * self.max_speed

        self.position += self.velocity * delta_t

class SwarmManager:
    def __init__(self, drones, interaction_radius=50.0, obstacles=None, safety_distance=10.0):
        self.drones = drones
        self.interaction_radius = interaction_radius
        self.obstacles = obstacles or []
        self.safety_distance = safety_distance

    def get_neighbors(self, drone):
        return [d for d in self.drones if d is not drone and np.linalg.norm(d.position - drone.position) < self.interaction_radius]

    def step(self, delta_t=0.1):
        neighbors_list = [self.get_neighbors(drone) for drone in self.drones]
        for drone, neighbors in zip(self.drones, neighbors_list):
            drone.update(neighbors, delta_t=delta_t, obstacles=self.obstacles, safety_distance=self.safety_distance)

            if self.environment is not None:
                (new_x, new_y), (new_vx, new_vy) = self.environment.resolve_boundary(drone.position, drone.velocity, delta_t=delta_t)
                drone.position = np.array([new_x, new_y], dtype=float)
                drone.velocity = np.array([new_vx, new_vy], dtype=float)
