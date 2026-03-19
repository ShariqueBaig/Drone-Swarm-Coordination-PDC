import pygame
import numpy as np
import config

def draw_obstacles(screen, obstacles):
    for x, y, r in obstacles:
        pygame.draw.circle(screen, (200, 50, 50), (int(x), int(y)), int(r))

def run_viz(swarm_mgr, env):
    """
    Entry point for the visualizer.
    swarm_mgr: SwarmManager instance
    env: Environment instance
    """
    main(swarm_mgr, env)

def main(swarm_mgr, env):
    pygame.init()
    screen = pygame.display.set_mode((env.width, env.height))
    pygame.display.set_caption("Drone Swarm Simulation (Milestone 1)")
    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                # Left click to add
                if event.button == 1:
                    env.obstacles.append((x, y, 20))
                # Right click to remove
                elif event.button == 3:
                    for i, (ox, oy, r) in enumerate(env.obstacles[:]):
                        if np.sqrt((x - ox)**2 + (y - oy)**2) <= r:
                            env.obstacles.pop(i)
                            break

        # Update Swarm
        swarm_mgr.update(config.dt)

        # Draw Frame
        screen.fill((30, 30, 35)) # Dark theme
        
        # Draw Obstacles (using Suffiyan's A1.2 list)
        draw_obstacles(screen, env.obstacles)

        # Draw Drones (Sharique's B1.1 vectorized data + heading lines)
        positions = swarm_mgr.positions
        velocities = swarm_mgr.velocities
        for i in range(len(positions)):
            pos = positions[i]
            vel = velocities[i]
            
            # 1. Draw the boid body
            pygame.draw.circle(screen, (100, 200, 255), pos.astype(int), 4)
            
            # 2. Draw the heading line (direction)
            speed = np.linalg.norm(vel)
            if speed > 0:
                direction = vel / speed
                end_pos = pos + direction * 10 # 10px heading line
                pygame.draw.line(screen, (255, 255, 255), pos.astype(int), end_pos.astype(int), 2)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
