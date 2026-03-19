import numpy as np

class CentralController:
    def __init__(self, drones, width, height, num_targets=20, max_speed=2.0, safety_distance=10.0):
        self.drones = drones
        self.width = width
        self.height = height
        self.num_targets = num_targets
        self.max_speed = max_speed
        self.safety_distance = safety_distance
        self.targets = self._generate_targets()
        self.assignment = [None] * len(drones)

    def _generate_targets(self):
        return [np.array([np.random.uniform(50, self.width - 50), np.random.uniform(50, self.height - 50)]) for _ in range(self.num_targets)]

    def assign_targets(self):
        for i, drone in enumerate(self.drones):
            self.assignment[i] = self.targets[i % len(self.targets)]

    def compute_velocities(self):
        velocities = []
        for i, drone in enumerate(self.drones):
            target = self.assignment[i]
            direction = target - drone.position
            dist = np.linalg.norm(direction)
            if dist > 0:
                direction = direction / dist
            obs_avoid = np.zeros(2)
            for ox, oy, orad in drone.obstacles if hasattr(drone, 'obstacles') else []:
                obs_pos = np.array([ox, oy])
                dd = np.linalg.norm(drone.position - obs_pos)
                if 0 < dd < orad + self.safety_distance * 2:
                    obs_avoid += (drone.position - obs_pos) / (dd + 1e-5)
            collision_avoid = np.zeros(2)
            for other in self.drones:
                if other is not drone:
                    dd = np.linalg.norm(drone.position - other.position)
                    if 0 < dd < self.safety_distance * 2:
                        collision_avoid += (drone.position - other.position) / (dd + 1e-5)
            combined = direction * 0.6 + obs_avoid * 0.3 + collision_avoid * 0.3
            if np.linalg.norm(combined) > 0:
                velocities.append(combined / np.linalg.norm(combined) * self.max_speed)
            else:
                velocities.append(np.zeros(2))
        return velocities

    def update(self):
        velocities = self.compute_velocities()
        for drone, vel in zip(self.drones, velocities):
            drone.velocity = vel
            drone.position += drone.velocity
