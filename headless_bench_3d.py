#!/usr/bin/env python3
"""
headless_bench_3d.py — Benchmark all 3 spatial algorithms (Octree / Grid / Naive)
using the 3D SwarmManager (simulation3d physics) without any rendering.

Outputs a timestamped CSV to benchmarks/ and prints a summary table.
Designed to run inside Docker (no GPU/display required).
"""
import time, sys, os, csv, json
import numpy as np

# Add Milestone 2 src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Milestone 2', 'src'))

import config
from swarm_3d import SwarmManager3D
from environment3d import Environment3D

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

FRAMES    = 600        # frames per algorithm
WARMUP    = 30         # warm-up frames (excluded from stats)
ALGORITHMS = ['octree', 'grid', 'naive']

def bench_one(algo_name, frames=FRAMES, warmup=WARMUP):
    """Run one algorithm headlessly and return stats dict."""
    env   = Environment3D()
    swarm = SwarmManager3D(env)
    swarm.set_method(algo_name)

    times = []
    cpu_vals = []
    mem_vals = []
    proc = psutil.Process() if HAS_PSUTIL else None

    print(f"\n{'='*60}")
    print(f"  BENCHMARKING: {algo_name.upper()}  ({frames} frames, {warmup} warmup)")
    print(f"{'='*60}")

    for i in range(frames):
        t0 = time.perf_counter()
        swarm.update()
        dt = time.perf_counter() - t0

        if i >= warmup:
            times.append(dt)
            if proc:
                try:
                    cpu_vals.append(proc.cpu_percent(interval=None))
                    mem_vals.append(proc.memory_info().rss / 1024 / 1024)
                except:
                    pass

        if (i + 1) % 100 == 0:
            recent = times[-100:] if times else [dt]
            avg = np.mean(recent) * 1000
            print(f"  Frame {i+1:5d} | avg frame {avg:.2f} ms | {1000/avg:.0f} FPS")

    arr = np.array(times) * 1000  # ms
    fps_arr = 1000.0 / arr

    stats = {
        'algorithm': algo_name.upper(),
        'total_frames': len(times),
        'fps_avg': float(np.mean(fps_arr)),
        'fps_median': float(np.median(fps_arr)),
        'fps_min': float(np.min(fps_arr)),
        'fps_max': float(np.max(fps_arr)),
        'fps_std': float(np.std(fps_arr)),
        'frame_time_avg_ms': float(np.mean(arr)),
        'frame_time_p50_ms': float(np.median(arr)),
        'frame_time_p95_ms': float(np.percentile(arr, 95)),
        'frame_time_p99_ms': float(np.percentile(arr, 99)),
        'frame_time_min_ms': float(np.min(arr)),
        'frame_time_max_ms': float(np.max(arr)),
        'cpu_avg': float(np.mean(cpu_vals)) if cpu_vals else 0.0,
        'cpu_peak': float(np.max(cpu_vals)) if cpu_vals else 0.0,
        'memory_avg_mb': float(np.mean(mem_vals)) if mem_vals else 0.0,
        'memory_peak_mb': float(np.max(mem_vals)) if mem_vals else 0.0,
        'coverage_pct': float(swarm.coverage_pct),
        'collisions': int(swarm.collision_count),
    }

    print(f"\n  Result:  {stats['fps_avg']:.1f} avg FPS  |  {stats['frame_time_avg_ms']:.2f} ms avg  |  p95 {stats['frame_time_p95_ms']:.2f} ms")
    return stats


def main():
    from datetime import datetime
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    os.makedirs('benchmarks', exist_ok=True)

    all_stats = []
    for algo in ALGORITHMS:
        stats = bench_one(algo)
        all_stats.append(stats)

    # ── Summary Table ─────────────────────────────────────────────────────
    print(f"\n\n{'='*80}")
    print(f"  HEADLESS 3D BENCHMARK SUMMARY  —  {config.num_boids} drones  |  {FRAMES - WARMUP} frames/algo")
    print(f"{'='*80}")
    print(f"  {'Algorithm':<12} {'Avg FPS':>10} {'Med FPS':>10} {'Min FPS':>10} {'Avg ms':>10} {'P95 ms':>10} {'CPU%':>8} {'Mem MB':>8}")
    print(f"  {'-'*78}")
    for s in all_stats:
        print(f"  {s['algorithm']:<12} {s['fps_avg']:>10.1f} {s['fps_median']:>10.1f} {s['fps_min']:>10.1f} {s['frame_time_avg_ms']:>10.2f} {s['frame_time_p95_ms']:>10.2f} {s['cpu_avg']:>8.1f} {s['memory_avg_mb']:>8.1f}")
    print(f"{'='*80}\n")

    # ── Save CSV ──────────────────────────────────────────────────────────
    csv_file = f'benchmarks/headless3d_benchmark_{ts}.csv'
    with open(csv_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=all_stats[0].keys())
        writer.writeheader()
        writer.writerows(all_stats)
    print(f"  CSV saved: {csv_file}")

    # ── Save JSON ─────────────────────────────────────────────────────────
    json_file = f'benchmarks/headless3d_benchmark_{ts}.json'
    with open(json_file, 'w') as f:
        json.dump({
            'timestamp': ts,
            'num_drones': config.num_boids,
            'frames_per_algo': FRAMES - WARMUP,
            'warmup_frames': WARMUP,
            'results': all_stats
        }, f, indent=2)
    print(f"  JSON saved: {json_file}\n")


if __name__ == '__main__':
    main()
