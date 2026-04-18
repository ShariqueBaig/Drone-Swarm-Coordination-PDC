"""
headless_bench.py — measures physics FPS without any rendering.
Runs 300 frames and prints average FPS.
"""
import time, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from environment import Environment
from swarm_optimized import SwarmManagerOptimized as SwarmManager
import config as _cfg

# Patch M2 collision steer
import numpy as np
env   = Environment("config.yaml")
swarm = SwarmManager(env)

_original_update = SwarmManager.update
_cache = [None, 0]

def _m2_update(self, dt=None):
    if dt is None: dt = _cfg.dt
    self.env.reload_if_changed()
    self.env.step(dt)
    _original_update(self, dt)
    if not hasattr(self.env, 'collision_steer'): return
    _cache[1] += 1
    if _cache[0] is None or _cache[1] >= 2:
        _cache[0] = self.env.collision_steer(
            positions=self.positions, velocities=self.velocities,
            max_speed=_cfg.max_speed, max_force=_cfg.max_force,
            pred_lookahead=4)
        _cache[1] = 0
    alive = ~self.dead_mask
    self.velocities[alive] += _cache[0][alive]
    spds = np.linalg.norm(self.velocities, axis=1)
    ov   = alive & (spds > _cfg.max_speed)
    self.velocities[ov] = (self.velocities[ov] / spds[ov, np.newaxis]) * _cfg.max_speed

SwarmManager.update = _m2_update

FRAMES = 300
times  = []
print(f"Benchmarking {FRAMES} frames (headless)...")
for i in range(FRAMES):
    t0 = time.perf_counter()
    swarm.update(_cfg.dt)
    times.append(time.perf_counter() - t0)

avg_ms  = sum(times) / len(times) * 1000
fps_sim = 1.0 / (sum(times) / len(times))
p95_ms  = sorted(times)[int(len(times)*0.95)] * 1000

print(f"\n=== HEADLESS PHYSICS BENCHMARK (300 frames) ===")
print(f"  Avg frame time : {avg_ms:.2f} ms")
print(f"  Simulated FPS  : {fps_sim:.1f}")
print(f"  95th pct       : {p95_ms:.2f} ms")
print(f"  Min/Max ms     : {min(times)*1000:.2f} / {max(times)*1000:.2f}")
