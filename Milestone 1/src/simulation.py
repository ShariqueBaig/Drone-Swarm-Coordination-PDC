import pygame
import sys
import config
from flock import Flock

def main():
    pygame.init()
    # screen_info = pygame.display.Info()
    # width, height = screen_info.current_w, screen_info.current_h
    # config.width = width
    # config.height = height
    
    screen = pygame.display.set_mode((config.width, config.height))
    pygame.display.set_caption("Boids Simulation (Vectorized)")
    clock = pygame.time.Clock()

    flock = Flock()

    font = pygame.font.SysFont("Arial", 18)

    running = True
    while running:
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Left click to add obstacle
                if event.button == 1:
                    x, y = pygame.mouse.get_pos()
                    flock.add_obstacle(x, y)
                # Right click to remove obstacle
                elif event.button == 3:
                    x, y = pygame.mouse.get_pos()
                    flock.remove_obstacle(x, y)
            elif event.type == pygame.KEYDOWN:
                 if event.key == pygame.K_r:
                     flock = Flock() # Reset

        # Update
        flock.update(config.dt)

        # Draw
        screen.fill(config.BACKGROUND_COLOR)
        flock.draw(screen)
        
        # Display FPS
        fps = str(int(clock.get_fps()))
        fps_text = font.render(f"FPS: {fps} | Boids: {config.num_boids}", 1, pygame.Color("white"))
        screen.blit(fps_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
