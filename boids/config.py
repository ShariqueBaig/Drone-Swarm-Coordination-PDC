import numpy as np

# Screen dimensions
width, height = 1200, 800

# Boid parameters
num_boids = 100
max_speed = 5
max_force = 0.1
perception_radius = 50

# Rule weights
separation_weight = 1.5
alignment_weight = 1.0
cohesion_weight = 1.0
obstacle_weight = 5.0  # Weight for avoiding obstacles

# Colors
BACKGROUND_COLOR = (30, 30, 30)
BOID_COLOR = (0, 191, 255)     # Deep Sky Blue
OBSTACLE_COLOR = (255, 69, 0)  # Red-Orange

# Obstacle parameters
obstacle_radius = 40
