"""
main.py — PDC Project · Spring 2026
Wires Environment (Suffiyan) + SwarmManagerOptimized (Sharique/Ashhal) + Visualizer (Usman).

Milestone 2 additions:
  - env.step(dt)            called before swarm.update() to advance dynamic obstacles (A2.3)
  - env.reload_if_changed() called every tick for config hot-reload (A2.4)
  - collision_steer passed into swarm via env for A2.1/A2.2/A2.5 (see swarm_m2_patch below)
"""

import numpy as np

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
_col_steer_cache = None   # cached collision steer (recomputed every 2 ticks)
_col_cache_tick  = 0

def _m2_update(self, dt=None):
    global _col_steer_cache, _col_cache_tick
    if dt is None:
        dt = _cfg.dt

    # A2.4 — Hot-reload config
    self.env.reload_if_changed()

    # A2.3 — Advance dynamic obstacles
    self.env.step(dt)

    # Run original swarm update
    _original_update(self, dt)

    # A2.1/A2.2/A2.5 — Inject collision steer (cached every 2 ticks)
    if not hasattr(self.env, 'collision_steer'):
        return

    _col_cache_tick += 1
    if _col_steer_cache is None or _col_cache_tick >= 2:
        _col_steer_cache = self.env.collision_steer(
            positions   = self.positions,
            velocities  = self.velocities,
            max_speed   = _cfg.max_speed,
            max_force   = _cfg.max_force,
            pred_lookahead = 4,   # reduced from 8 — halves predictive cost
        )
        _col_cache_tick = 0

    alive = ~self.dead_mask
    self.velocities[alive] += _col_steer_cache[alive]

    # Re-clamp speed
    speeds = np.linalg.norm(self.velocities, axis=1)
    over   = alive & (speeds > _cfg.max_speed)
    self.velocities[over] = (
        self.velocities[over] / speeds[over, np.newaxis]
    ) * _cfg.max_speed

    # ── M2: task allocation + formation (B2.2–B2.5) ──────────────────────────
    comm_mask = self.neighbor_mask  # already built by find_neighbors_*
    if comm_mask is not None and hasattr(self, 'auction_tasks'):
        self.auction_tasks(comm_mask)
        task_s      = self.calculate_task_steer()
        formation_s = self.calculate_formation_steer()
        task_w      = getattr(_cfg, 'task_weight',      2.0)
        form_w      = getattr(_cfg, 'formation_weight', 0.5)
        self.velocities[alive] += (task_s[alive] * task_w +
                                   formation_s[alive] * form_w)
        # Final speed clamp
        speeds2 = np.linalg.norm(self.velocities, axis=1)
        over2   = alive & (speeds2 > _cfg.max_speed)
        self.velocities[over2] = (
            self.velocities[over2] / speeds2[over2, np.newaxis]
        ) * _cfg.max_speed

SwarmManager.update = _m2_update

# ── Run simulation ────────────────────────────────────────────────────────────
run_viz(swarm_mgr=swarm_manager, env=env)