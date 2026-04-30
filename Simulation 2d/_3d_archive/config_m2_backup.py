"""
config.py — Shared constants · PDC Project Spring 2026
Matches config.yaml for world dimensions and boundary_margin.

Changes from previous version:
  width/height      1200x800 → 1000x1000  (match config.yaml)
  boundary_margin   100 → 50              (less aggressive wall avoidance)
  separation_weight 1.5 → 2.5            (strong but not overpowering)
  safety_distance   20  → 25             (slightly earlier separation trigger)
  obstacle_weight   5.0 → 2.5            (was too strong — prevented maneuvering)
"""

import numpy as np

# ── World / screen ────────────────────────────────────────────────────────────
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
separation_weight = 2.5
alignment_weight  = 1.0
cohesion_weight   = 1.0
obstacle_weight   = 1.6       # reduced for tighter maneuverability
boundary_weight   = 1.0

# ── Colors ────────────────────────────────────────────────────────────────────
BACKGROUND_COLOR = (30, 30, 30)
BOID_COLOR       = (0, 191, 255)
OBSTACLE_COLOR   = (255, 69, 0)

# ── Obstacle / boundary ───────────────────────────────────────────────────────
obstacle_radius = 40
boundary_margin = 50       # velocity flip fires this far from wall

# ── Optimization ─────────────────────────────────────────────────────────────
cell_size     = perception_radius
log_frequency = 100

# ── Milestone 2 ───────────────────────────────────────────────────────────────
task_weight = 2.0
formation_weight = 1.0
communication_radius = 100
task_radius = 20
