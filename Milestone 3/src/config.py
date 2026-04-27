"""
config.py — Shared constants · PDC Project Spring 2026 · Milestone 3
Matches config.yaml for world dimensions and boundary_margin.

Milestone 3 additions:
  - Parallelism configuration (num_threads, use_gpu, pipeline, cache_line)
  - Amdahl's law measurement toggle
  - GPU device selection for RTX 4050
"""

import numpy as np
import os

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

# ═══════════════════════════════════════════════════════════════════════════════
#  PDC TECHNIQUE: Configuration — Milestone 3 Parallelism Settings
# ═══════════════════════════════════════════════════════════════════════════════

# ── Thread / Process Parallelism ──────────────────────────────────────────────
num_threads       = min(os.cpu_count() or 4, 8)   # Worker threads for Fork-Join
pipeline_enabled  = True                            # Pipeline-parallel double buffering
cache_line_size   = 64                              # Bytes; for false-sharing avoidance padding

# ── GPU / GPGPU (CUDA via CuPy) ──────────────────────────────────────────────
# Set use_gpu = True to enable GPU acceleration (requires cupy + NVIDIA CUDA)
# Fully utilises RTX 4050 (Ada Lovelace, Compute Capability 8.9, 6GB VRAM)
use_gpu           = True     # Will auto-fallback to False if cupy unavailable
gpu_device_id     = 0        # CUDA device index (0 = primary GPU)
gpu_block_size    = 256      # CUDA threads per block (optimal for RTX 4050)
gpu_stream_count  = 2        # Concurrent CUDA streams for async kernel overlap

# ── Amdahl's / Gustafson's Law Measurement ────────────────────────────────────
enable_timing     = True     # Enable per-technique timing instrumentation
timing_warmup     = 50       # Frames to skip before timing (JIT warmup)
