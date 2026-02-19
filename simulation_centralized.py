import pygame
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 800, 600
NUM_DRONES = 100
DRONE_RADIUS = 5
SAFETY_DISTANCE = 10
DELTA_T = 0.1
MAX_SPEED = 2
OBSTACLES = [(200, 300, 50), (600, 400, 30)]  # list of (x, y, radius)

class Drone:
    def __init__(self, x, y, drone_id):
        self.position = np.array([float(x), float(y)])
        self.velocity = np.array([0.0, 0.0])
        self.drill_id = drone_id
        self.assigned_target = None

    def update(self, velocity):
        """Update drone position with centrally assigned velocity"""
        self.velocity = velocity
        self.position += self.velocity * DELTA_T
        
        # Keep in bounds with bounce
        if self.position[0] <= 0 or self.position[0] >= WIDTH:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], 0, WIDTH)
        if self.position[1] <= 0 or self.position[1] >= HEIGHT:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], 0, HEIGHT)


class CentralController:
    """Centralized controller that manages all drones"""
    def __init__(self, drones):
        self.drones = drones
        self.targets = self._generate_targets()
        self.assignment = [None] * len(drones)
    
    def _generate_targets(self):
        """Generate 20 target points for drones to visit"""
        targets = []
        for _ in range(20):
            x = random.randint(50, WIDTH - 50)
            y = random.randint(50, HEIGHT - 50)
            targets.append(np.array([float(x), float(y)]))
        return targets
    
    def assign_targets(self):
        """Assign targets to drones (simple round-robin)"""
        for i, drone in enumerate(self.drones):
            target_idx = i % len(self.targets)
            self.assignment[i] = self.targets[target_idx]
    
    def compute_velocities(self):
        """Compute velocities for all drones"""
        velocities = []
        
        for i, drone in enumerate(self.drones):
            target = self.assignment[i]
            
            # Direction to target
            direction = target - drone.position
            dist_to_target = np.linalg.norm(direction)
            
            if dist_to_target < 5:
                # Target reached, pick next target
                target_idx = (i + 1) % len(self.targets)
                self.assignment[i] = self.targets[target_idx]
                target = self.assignment[i]
                direction = target - drone.position
                dist_to_target = np.linalg.norm(direction)
            
            if dist_to_target > 0:
                direction = direction / dist_to_target
            
            # Obstacle avoidance
            obs_avoid = np.array([0.0, 0.0])
            for obs_x, obs_y, obs_r in OBSTACLES:
                obs_pos = np.array([obs_x, obs_y])
                dist_to_obs = np.linalg.norm(drone.position - obs_pos)
                if 0 < dist_to_obs < obs_r + SAFETY_DISTANCE * 2:
                    avoid_vec = drone.position - obs_pos
                    obs_avoid += avoid_vec / (dist_to_obs + 0.1)
            
            # Collision avoidance with other drones
            collision_avoid = np.array([0.0, 0.0])
            for other in self.drones:
                if other != drone:
                    dist = np.linalg.norm(drone.position - other.position)
                    if 0 < dist < SAFETY_DISTANCE * 2:
                        avoid_vec = drone.position - other.position
                        collision_avoid += avoid_vec / (dist + 0.1)
            
            # Combine all forces
            combined = direction * 0.6 + obs_avoid * 0.3 + collision_avoid * 0.3
            
            if np.linalg.norm(combined) > 0:
                velocity = combined / np.linalg.norm(combined) * MAX_SPEED
            else:
                velocity = np.array([0.0, 0.0])
            
            velocities.append(velocity)
        
        return velocities
    
    def update(self):
        """Compute and assign new velocities for all drones"""
        velocities = self.compute_velocities()
        for drone, vel in zip(self.drones, velocities):
            drone.update(vel)


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Centralized Drone Swarm Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    drones = [Drone(random.randint(0, WIDTH), random.randint(0, HEIGHT), i) for i in range(NUM_DRONES)]
    controller = CentralController(drones)
    controller.assign_targets()

    running = True
    frame = 0
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill((0, 0, 0))

        # Draw obstacles
        for obs_x, obs_y, obs_r in OBSTACLES:
            pygame.draw.circle(screen, (255, 0, 0), (obs_x, obs_y), obs_r)
        
        # Draw targets
        for i, target in enumerate(controller.targets):
            pygame.draw.circle(screen, (0, 255, 0), target.astype(int), 4)

        # Update drones via central controller
        controller.update()
        
        # Draw drones
        for drone in drones:
            pygame.draw.circle(screen, (255, 255, 255), drone.position.astype(int), DRONE_RADIUS)
        
        # Display stats
        text = font.render(f"Drones: {NUM_DRONES} | Centralized Control | Frame: {frame}", True, (100, 200, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()

if __name__ == "__main__":
    main()
