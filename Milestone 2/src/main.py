"""
main.py — PDC Project · Spring 2026
Wires Environment (Suffiyan) + SwarmManagerOptimized (Sharique/Ashhal) + Visualizer (Usman).

Milestone 2 additions:
  - env.step(dt)            called before swarm.update() to advance dynamic obstacles (A2.3)
  - env.reload_if_changed() called every tick for config hot-reload (A2.4)
  - collision_steer passed into swarm via env for A2.1/A2.2/A2.5 (see swarm_m2_patch below)
"""

try:
    from environment import Environment
    from swarm_optimized import SwarmManagerOptimized as SwarmManager
    from visualizer import run_viz
except ImportError as e:
    print(f"Import Error: {e}")
    import sys
    sys.exit(1)

# ── Initialize ────────────────────────────────────────────────────────────────
env           = Environment("config.yaml")
swarm_manager = SwarmManager(env)

# ── Monkey-patch M2 collision steer into the swarm update loop ────────────────
# This keeps Ashhal's swarm_optimized.py untouched while injecting
# Suffiyan's A2.1/A2.2/A2.5 forces into the acceleration pipeline.
#
# How it works:
#   1. We wrap SwarmManager.update() so env.step() and env.reload_if_changed()
#      always fire first.
#   2. After the original update() runs, we inject the collision_steer force
#      into self.accelerations (it's added as a post-step correction so it
#      doesn't interfere with Ashhal's force ordering).

import config as _cfg

_original_update = SwarmManager.update

def _m2_update(self, dt=None):
    if dt is None:
        dt = _cfg.dt

    # A2.4 — Hot-reload config before each tick
    self.env.reload_if_changed()

    # A2.3 — Advance dynamic obstacles before swarm physics
    self.env.step(dt)

    # Run original swarm update (Ashhal's full pipeline)
    _original_update(self, dt)

    # A2.1 / A2.2 / A2.5 — Inject Suffiyan's collision steer post-update
    # (adds to already-integrated velocity so it acts as a velocity correction
    #  rather than competing inside the force accumulator)
    alive = ~self.dead_mask
    if not hasattr(self.env, 'collision_steer'):
        return  # safety guard

    col_steer = self.env.collision_steer(
        positions       = self.positions,
        velocities      = self.velocities,
        max_speed       = _cfg.max_speed,
        max_force       = _cfg.max_force,
    )
    self.velocities[alive] += col_steer[alive]

    # Re-clamp speed after correction
    import numpy as np
    speeds = np.linalg.norm(self.velocities, axis=1)
    over   = alive & (speeds > _cfg.max_speed)
    self.velocities[over] = (
        self.velocities[over] / speeds[over, np.newaxis]
    ) * _cfg.max_speed

SwarmManager.update = _m2_update

# ── Run simulation ────────────────────────────────────────────────────────────
run_viz(swarm_mgr=swarm_manager, env=env)