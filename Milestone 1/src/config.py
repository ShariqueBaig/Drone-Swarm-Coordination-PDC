import numpy as np

# Screen dimensions
width, height = 1200, 800

# Boid parameters
num_boids = 100
max_speed = 250         # Scaled up for dt=0.02 (was 5 per frame)
max_force = 5.0         # Scaled up for dt=0.02 (was 0.1 per frame)
perception_radius = 50
safety_distance = 20    # Radius for reactive avoidance (separation)
seed = 42               # Random initialization seed
dt = 0.02               # Fixed time step

# Rule weights
separation_weight = 1.5
alignment_weight = 1.0
cohesion_weight = 1.0
obstacle_weight = 5.0  # Weight for avoiding obstacles
boundary_weight = 1.0  # Weight for staying within bounds

# Colors
BACKGROUND_COLOR = (30, 30, 30)
BOID_COLOR = (0, 191, 255)     # Deep Sky Blue
OBSTACLE_COLOR = (255, 69, 0)  # Red-Orange

# Obstacle parameters
obstacle_radius = 40
boundary_margin = 100 # Distance from wall to start steering away
