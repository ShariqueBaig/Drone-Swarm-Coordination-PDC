"""
benchmark_runner.py — Structured method comparison
PDC Project · Spring 2026

PURPOSE: Run naive / grid / quadtree for FRAMES_PER_METHOD frames each,
         discard WARMUP_FRAMES per method, log to separate CSVs per method
         and one combined CSV for visualization.

WHY THIS IS NEEDED:
  The original approach (switching method mid-run) has warmup bias:
  - 'naive' starts cold  → low early FPS inflates its apparent cost
  - 'quadtree' starts after Python is warmed up → looks faster than it is
  A fair comparison must give each method the same warmup budget.

USAGE:
    python benchmark_runner.py
    (no pygame window — headless physics-only benchmark)
"""

import numpy as np
import csv
import time
import psutil
from collections import deque
from datetime import datetime

FRAMES_PER_METHOD = 500
WARMUP_FRAMES     = 50
METHODS           = ['naive', 'grid', 'quadtree']
OUTPUT_CSV        = 'benchmark_comparison.csv'


def make_env():
    """Minimal environment — no pygame required."""
    class Env:
        width    = 1000
        height   = 1000
        boundary = 'hard-wall'
        boundary_margin = 50
        boundary_repulsion_strength = 1000.0
        dt       = 0.02
        seed     = 42
        num_drones = 100
        obstacles = [(200, 300, 50), (600, 400, 30), (350, 200, 45), (700, 650, 60)]

        _obs_dirty = True
        _obs_centers_cache = None
        _obs_radii_cache   = None

        def _rebuild_obstacle_cache(self):
            import numpy as np
            self._obs_centers_cache = np.array(
                [[o[0], o[1]] for o in self.obstacles], dtype=np.float64)
            self._obs_radii_cache = np.array(
                [o[2] for o in self.obstacles], dtype=np.float64)
            self._obs_dirty = False

        @property
        def obs_centers(self):
            if self._obs_dirty: self._rebuild_obstacle_cache()
            return self._obs_centers_cache

        @property
        def obs_radii(self):
            if self._obs_dirty: self._rebuild_obstacle_cache()
            return self._obs_radii_cache

        def add_obstacle(self, x, y, r=20):
            self.obstacles.append((x, y, r))
            self._obs_dirty = True

        def remove_obstacle(self, i):
            self.obstacles.pop(i)
            self._obs_dirty = True

        def resolve_boundary_batch(self, positions, velocities, dt=0.02):
            import numpy as np
            left  = positions[:, 0] < 0
            right = positions[:, 0] > self.width
            top   = positions[:, 1] < 0
            bot   = positions[:, 1] > self.height
            positions[left,  0] = 0;     velocities[left,  0] *= -1
            positions[right, 0] = self.width; velocities[right, 0] *= -1
            positions[top,   1] = 0;     velocities[top,   1] *= -1
            positions[bot,   1] = self.height; velocities[bot, 1] *= -1
            return positions, velocities

    e = Env()
    e._rebuild_obstacle_cache()
    return e


def run_benchmark():
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from swarm_optimized import SwarmManagerOptimized

    # Prime psutil
    psutil.cpu_percent(interval=None)

    rows = []

    for method in METHODS:
        print(f"\n{'='*50}")
        print(f"  Benchmarking method: {method}")
        print(f"  Warmup: {WARMUP_FRAMES} frames | Measure: {FRAMES_PER_METHOD} frames")
        print(f"{'='*50}")

        env = make_env()
        swarm = SwarmManagerOptimized(env)
        swarm.set_method(method)

        frame_times = deque(maxlen=60)
        measured    = 0

        for frame in range(FRAMES_PER_METHOD + WARMUP_FRAMES):
            t0 = time.perf_counter()
            swarm.update(env.dt)
            elapsed = time.perf_counter() - t0

            frame_times.append(elapsed)

            # Skip warmup
            if frame < WARMUP_FRAMES:
                continue

            measured += 1
            avg_ft = sum(frame_times) / len(frame_times)
            fps    = 1.0 / avg_ft if avg_ft > 0 else 0.0

            if measured % 5 == 0:
                cpu = psutil.cpu_percent(interval=None)
                mem = psutil.Process().memory_info().rss / 1024 / 1024
                rows.append({
                    'timestamp':    datetime.now().isoformat(),
                    'method':       method,
                    'frame':        measured,
                    'fps':          round(fps, 2),
                    'cpu_percent':  round(cpu, 1),
                    'memory_mb':    round(mem, 1),
                    'avg_neighbors':round(float(swarm.avg_neighbors), 2),
                    'num_drones':   swarm.num_boids,
                })

            if measured % 100 == 0:
                fps_str = f"{fps:.1f}" if fps < 9999 else ">9999"
                print(f"  [{method}] frame {measured}/{FRAMES_PER_METHOD}: "
                      f"{fps_str} FPS | neighbors={swarm.avg_neighbors:.1f}")

        method_fps = [r['fps'] for r in rows if r['method'] == method]
        print(f"\n  {method} summary: "
              f"mean={np.mean(method_fps):.1f} "
              f"median={np.median(method_fps):.1f} "
              f"min={np.min(method_fps):.1f} "
              f"max={np.max(method_fps):.1f} FPS")

    # Write combined CSV
    fieldnames = ['timestamp', 'method', 'frame', 'fps', 'cpu_percent',
                  'memory_mb', 'avg_neighbors', 'num_drones']
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBenchmark complete. Results saved to: {OUTPUT_CSV}")
    print("Run visualize_benchmark.py to generate comparison plots.")
    return OUTPUT_CSV


if __name__ == '__main__':
    run_benchmark()
