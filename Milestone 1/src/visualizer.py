import pygame
import numpy as np
import config

# C1.1 - C1.5: Usman's modular UI rendering loop
def main(swarm_mgr, environment_obstacles=None):
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Boids Simulation (Modular)")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = pygame.mouse.get_pos()
                if environment_obstacles is not None:
                    # Left click to add
                    if event.button == 1:
                        environment_obstacles.append(pygame.Rect(x-20, y-20, 40, 40))
                    # Right click to remove
                    elif event.button == 3:
                        for obs in environment_obstacles[:]:
                            if obs.collidepoint(x, y):
                                environment_obstacles.remove(obs)

        # Perform backend logic integrations
        swarm_mgr.obstacles = environment_obstacles if environment_obstacles else []
        swarm_mgr.update(config.dt)

        # Draw Frame
        screen.fill(config.BACKGROUND_COLOR)
        
        if environment_obstacles:
            for obs in environment_obstacles:
                pygame.draw.rect(screen, config.OBSTACLE_COLOR, obs)

        # Usman accesses the agreed B1.1 and B1.2 NumPy structure data here
        positions = swarm_mgr.positions
        velocities = swarm_mgr.velocities
        num_boids = len(positions)
        
        angles = np.arctan2(velocities[:, 1], velocities[:, 0])
        r = 6
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)
        
        p1_x = positions[:, 0] + cos_a * r * 2
        p1_y = positions[:, 1] + sin_a * r * 2
        
        cos_a_l = np.cos(angles + 2.5)
        sin_a_l = np.sin(angles + 2.5)
        p2_x = positions[:, 0] + cos_a_l * r
        p2_y = positions[:, 1] + sin_a_l * r
        
        cos_a_r = np.cos(angles - 2.5)
        sin_a_r = np.sin(angles - 2.5)
        p3_x = positions[:, 0] + cos_a_r * r
        p3_y = positions[:, 1] + sin_a_r * r
        
        p1 = np.stack((p1_x, p1_y), axis=1)
        p2 = np.stack((p2_x, p2_y), axis=1)
        p3 = np.stack((p3_x, p3_y), axis=1)
        
        line_length = 20
        line_end_x = p1_x + cos_a * line_length
        line_end_y = p1_y + sin_a * line_length
        line_end = np.stack((line_end_x, line_end_y), axis=1)

        for i in range(num_boids):
             pygame.draw.polygon(screen, config.BOID_COLOR, [p1[i], p2[i], p3[i]])
             pygame.draw.line(screen, (200, 200, 200), p1[i], line_end[i], 1)
        
        fps = str(int(clock.get_fps()))
        fps_text = font.render(f"FPS: {fps} | Modulated Architecture", 1, pygame.Color("white"))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
