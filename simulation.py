import pygame
import numpy as np
import random

# Constants
WIDTH, HEIGHT = 800, 600
NUM_DRONES = 100
DRONE_RADIUS = 5
INTERACTION_RADIUS = 50
SAFETY_DISTANCE = 10
DELTA_T = 0.1
MAX_SPEED = 30
OBSTACLES = [(200, 300, 50), (600, 400, 30)]  # list of (x, y, radius)
SEPARATION_WEIGHT = 1.5
ALIGNMENT_WEIGHT = 1.0
COHESION_WEIGHT = 1.0

class Drone:
    def __init__(self, x, y):
        self.position = np.array([float(x), float(y)])
        self.velocity = np.array([random.uniform(-1, 1), random.uniform(-1, 1)])
        if np.linalg.norm(self.velocity) > 0:
            self.velocity = self.velocity / np.linalg.norm(self.velocity) * MAX_SPEED
        else:
            self.velocity = np.array([0.0, 0.0])
    
    def separation(self, neighbors):
        """Steer to avoid crowding local flockmates"""
        steer = np.array([0.0, 0.0])
        if len(neighbors) == 0:
            return steer
        
        for neighbor in neighbors:
            diff = self.position - neighbor.position
            dist = np.linalg.norm(diff)
            if dist > 0:
                steer += diff / (dist + 0.1)
        
        if np.linalg.norm(steer) > 0:
            steer = steer / np.linalg.norm(steer) * MAX_SPEED
        return steer
    
    def alignment(self, neighbors):
        """Steer towards the average heading of local flockmates"""
        avg_vel = np.array([0.0, 0.0])
        if len(neighbors) == 0:
            return avg_vel
        
        for neighbor in neighbors:
            avg_vel += neighbor.velocity
        
        avg_vel = avg_vel / len(neighbors)
        if np.linalg.norm(avg_vel) > 0:
            avg_vel = avg_vel / np.linalg.norm(avg_vel) * MAX_SPEED
        return avg_vel
    
    def cohesion(self, neighbors):
        """Steer to move toward the average location of local flockmates"""
        center = np.array([0.0, 0.0])
        if len(neighbors) == 0:
            return center
        
        for neighbor in neighbors:
            center += neighbor.position
        center = center / len(neighbors)
        
        steer = center - self.position
        if np.linalg.norm(steer) > 0:
            steer = steer / np.linalg.norm(steer) * MAX_SPEED
        return steer

    def update(self, neighbors):
        """Update drone position using flocking behavior and obstacle avoidance"""
        # Flocking forces
        sep = self.separation(neighbors)
        ali = self.alignment(neighbors)
        coh = self.cohesion(neighbors)
        
        # Combine forces with weights
        combined_force = (sep * SEPARATION_WEIGHT + 
                         ali * ALIGNMENT_WEIGHT + 
                         coh * COHESION_WEIGHT)
        
        # Obstacle avoidance force
        obs_force = np.array([0.0, 0.0])
        for obs_x, obs_y, obs_r in OBSTACLES:
            obs_pos = np.array([obs_x, obs_y])
            dist = np.linalg.norm(self.position - obs_pos)
            if 0 < dist < obs_r + SAFETY_DISTANCE * 2:
                obs_vec = self.position - obs_pos
                obs_force += obs_vec / (dist + 0.1)
        
        if np.linalg.norm(obs_force) > 0:
            obs_force = obs_force / np.linalg.norm(obs_force) * MAX_SPEED * 2
        
        # Combine flocking and obstacle avoidance, prioritizing obstacles
        final_force = combined_force + obs_force * 2
        
        if np.linalg.norm(final_force) > 0:
            self.velocity = final_force / np.linalg.norm(final_force) * MAX_SPEED
        
        self.position += self.velocity * DELTA_T
        
        # Keep in bounds with bounce
        if self.position[0] <= 0 or self.position[0] >= WIDTH:
            self.velocity[0] *= -1
            self.position[0] = np.clip(self.position[0], 0, WIDTH)
        if self.position[1] <= 0 or self.position[1] >= HEIGHT:
            self.velocity[1] *= -1
            self.position[1] = np.clip(self.position[1], 0, HEIGHT)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Drone Swarm Simulation - Flocking Behavior")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 24)

    drones = [Drone(random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(NUM_DRONES)]

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

        for drone in drones:
            neighbors = [d for d in drones if np.linalg.norm(drone.position - d.position) < INTERACTION_RADIUS and d != drone]
            drone.update(neighbors)
            pygame.draw.circle(screen, (255, 255, 255), drone.position.astype(int), DRONE_RADIUS)
        
        # Display stats
        text = font.render(f"Drones: {NUM_DRONES} | Frame: {frame}", True, (100, 200, 255))
        screen.blit(text, (10, 10))

        pygame.display.flip()
        clock.tick(60)
        frame += 1

    pygame.quit()

if __name__ == "__main__":
    main()