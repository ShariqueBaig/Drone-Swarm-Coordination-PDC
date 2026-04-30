"""
config.py — Shared constants · PDC Project Spring 2026
2D Drone Swarm Coordination Simulation

Parameters for boid rules, world dimensions, and PDC settings.
"""

import os
import numpy as np

# ── World ─────────────────────────────────────────────────────────────────────
width  = 1000
height = 1000

# ── Boid parameters ───────────────────────────────────────────────────────────
num_boids         = 100
max_speed         = 250
max_force         = 5.0
perception_radius = 50
safety_distance   = 25
seed              = 42
dt                = 0.02

# ── Rule weights ──────────────────────────────────────────────────────────────
separation_weight = 12.0      # Strong repulsion to prevent overlap
alignment_weight  = 1.0
cohesion_weight   = 0.6       # Reduced to allow spreading for coverage
obstacle_weight   = 1.6
boundary_weight   = 2.5       # Strong to prevent wall clipping

# ── Colors ────────────────────────────────────────────────────────────────────
BACKGROUND_COLOR = (30, 30, 30)
BOID_COLOR       = (0, 191, 255)
OBSTACLE_COLOR   = (255, 69, 0)

# ── Obstacle / boundary ──────────────────────────────────────────────────────
obstacle_radius = 40
boundary_margin = 50

# ── Optimization ─────────────────────────────────────────────────────────────
cell_size     = perception_radius
log_frequency = 100

# ── Mission / task ────────────────────────────────────────────────────────────
task_weight          = 2.0
formation_weight     = 1.0
communication_radius = 100
task_radius          = 20
waypoint_weight      = 2.5

# ── PDC: Thread Parallelism ──────────────────────────────────────────────────
num_threads      = min(os.cpu_count() or 4, 8)

# ── PDC: Amdahl's / Gustafson's Law Measurement ─────────────────────────────
enable_timing    = True
timing_warmup    = 50       # Frames to skip before timing (JIT warmup)
