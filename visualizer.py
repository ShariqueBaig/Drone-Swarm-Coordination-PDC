import pygame

def initialize_window(width, height, title="Drone Swarm"):
    pygame.init()
    screen = pygame.display.set_mode((int(width), int(height)))
    pygame.display.set_caption(title)
    return screen

def draw_obstacles(screen, obstacles, color=(255, 0, 0)):
    for ox, oy, r in obstacles:
        pygame.draw.circle(screen, color, (int(ox), int(oy)), int(r), width=0)

def draw_drones(screen, drones, color=(255, 255, 255), radius=4):
    for drone in drones:
        pygame.draw.circle(screen, color, drone.position.astype(int), radius)

def draw_text(screen, text, x, y, color=(100, 200, 255), size=20):
    font = pygame.font.Font(None, size)
    surface = font.render(text, True, color)
    screen.blit(surface, (x, y))

def clear_screen(screen, color=(0, 0, 0)):
    screen.fill(color)

def refresh(vis_clock, fps=60):
    pygame.display.flip()
    vis_clock.tick(fps)
