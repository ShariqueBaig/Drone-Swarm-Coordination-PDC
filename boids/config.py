import numpy as np

# Screen dimensions (Milestone 1 requirement)
width, height = 1000, 1000

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
boundary_weight = 1.0  # Weight for staying within bounds

# Boundary handling
wrap_around = False   # Set True for toroidal world (Milestone 1 option)
boundary_margin = 20  # Distance from wall to begin repulsion
boundary_repulsion_strength = 1000.0

# Colors
BACKGROUND_COLOR = (30, 30, 30)
BOID_COLOR = (0, 191, 255)     # Deep Sky Blue
OBSTACLE_COLOR = (255, 69, 0)  # Red-Orange

# Obstacle parameters
obstacle_radius = 40

# Static obstacles for Milestone 1
static_obstacles = [
    (200, 200),
    (350, 600),
    (500, 400),
    (700, 250),
    (800, 750)
]
