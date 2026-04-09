# run_visualization.py
import pygame
import sys
from environment import Environment
from swarm_optimized import SwarmManagerOptimized
import config

def run_viz(swarm_mgr, env):
    pygame.init()
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Drone Swarm Simulation - Optimized")
    clock = pygame.time.Clock()
    
    running = True
    paused = False
    frame_count = 0
    
    # Font for stats
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_n:
                    swarm_mgr.set_method('naive')
                elif event.key == pygame.K_g:
                    swarm_mgr.set_method('grid')
                elif event.key == pygame.K_q:
                    swarm_mgr.set_method('quadtree')
        
        if not paused:
            swarm_mgr.update()
            frame_count += 1
        
        # Clear screen
        screen.fill(config.BACKGROUND_COLOR)
        
        # Draw obstacles
        for obs in env.obstacles:
            pygame.draw.circle(screen, config.OBSTACLE_COLOR, 
                             (int(obs[0]), int(obs[1])), int(obs[2]))
        
        # Draw drones
        for i, pos in enumerate(swarm_mgr.positions):
            # Color based on neighbor count
            if swarm_mgr.neighbor_counts[i] > 0:
                color = (0, 255, 0)  # Green = has neighbors
            else:
                color = config.BOID_COLOR  # Blue = isolated
            
            pygame.draw.circle(screen, color, 
                             (int(pos[0]), int(pos[1])), 4)
        
        # Draw stats
        fps_text = small_font.render(f"FPS: {int(clock.get_fps())}", True, (255, 255, 255))
        method_text = small_font.render(f"Method: {swarm_mgr.use_method}", True, (255, 255, 255))
        neighbors_text = small_font.render(f"Avg Neighbors: {swarm_mgr.avg_neighbors:.2f}", True, (255, 255, 255))
        
        screen.blit(fps_text, (10, 10))
        screen.blit(method_text, (10, 35))
        screen.blit(neighbors_text, (10, 60))
        
        # Instructions
        inst1 = small_font.render("N: Naive | G: Grid | Q: Quadtree | SPACE: Pause", True, (200, 200, 200))
        screen.blit(inst1, (10, config.height - 30))
        
        pygame.display.flip()
        clock.tick(60)  # Limit to 60 FPS
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    env = Environment("config.yaml")
    swarm = SwarmManagerOptimized(env)
    
    # Start with grid method
    swarm.set_method('grid')
    
    run_viz(swarm, env)