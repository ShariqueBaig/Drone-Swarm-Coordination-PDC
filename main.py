import sys
import numpy as np
import pygame

from environment import Environment
from swarm import Drone, SwarmManager
from optimization import CentralController
from visualizer import initialize_window, draw_obstacles, draw_drones, draw_text, clear_screen, refresh


def run_decentralized(env, num_drones=100, mode="boids"):
    drones = [Drone(np.random.uniform(0, env.width), np.random.uniform(0, env.height), max_speed=5.0) for _ in range(num_drones)]
    manager = SwarmManager(drones, interaction_radius=50.0, obstacles=env.obstacles, safety_distance=10.0)

    screen = initialize_window(env.width, env.height, title="Decentralized Drone Swarm")
    clock = pygame.time.Clock()

    running = True
    frame = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clear_screen(screen, color=(0, 0, 0))
        draw_obstacles(screen, env.obstacles)

        manager.step(delta_t=0.1)
        draw_drones(screen, drones)

        draw_text(screen, f"Frame: {frame} | Drones: {len(drones)}", 10, 10)
        refresh(clock, fps=60)
        frame += 1

    pygame.quit()


def run_centralized(env, num_drones=100):
    drones = [Drone(np.random.uniform(0, env.width), np.random.uniform(0, env.height), max_speed=2.0) for _ in range(num_drones)]
    controller = CentralController(drones, width=env.width, height=env.height, num_targets=20, max_speed=2.0)
    controller.assign_targets()

    screen = initialize_window(env.width, env.height, title="Centralized Drone Swarm")
    clock = pygame.time.Clock()

    running = True
    frame = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        clear_screen(screen, color=(0, 0, 0))
        draw_obstacles(screen, env.obstacles)

        controller.update()
        draw_drones(screen, drones)

        draw_text(screen, f"Frame: {frame} | Mode: Centralized | Drones: {len(drones)}", 10, 10)
        refresh(clock, fps=60)
        frame += 1

    pygame.quit()


def main():
    env = Environment()
    mode = sys.argv[1] if len(sys.argv) > 1 else "decentralized"
    num_drones = int(sys.argv[2]) if len(sys.argv) > 2 else 80

    if mode == "centralized":
        run_centralized(env, num_drones=num_drones)
    else:
        run_decentralized(env, num_drones=num_drones)


if __name__ == "__main__":
    main()
