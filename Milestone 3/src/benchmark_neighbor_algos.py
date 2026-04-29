"""
benchmark_neighbor_algos.py — Pure Algorithm Performance Comparison
Milestone 3 · Isolated Neighbor-Finding Benchmark

Tests naive O(n²) vs KDTree/Octree algorithms across different drone scales.
Zero rendering overhead — pure computation timing.

Usage:
    python benchmark_neighbor_algos.py

Output:
    - Console: ASCII table with timing and speedup
    - CSV: neighbor_algo_benchmark.csv (for graphs/analysis)
"""

import numpy as np
import time
import csv
from environment3d import Environment3D
from swarm_3d import SwarmManager3D
import config


def run_benchmark(drone_counts=[100, 250, 500, 1000, 2000]):
    """
    Benchmark naive vs octree algorithms at different scales.
    
    Returns:
        results: List of dicts with drone_count, naive_time, octree_time, speedup
    """
    results = []
    
    print("\n" + "="*80)
    print("  NEIGHBOR-FINDING ALGORITHM BENCHMARK")
    print("  Naive O(n²) vs KDTree/Octree")
    print("="*80)
    print(f"  Perception Radius: {config.perception_radius}")
    print(f"  Warmup Frames: 10 (JIT/cache stabilization)")
    print(f"  Measurement Frames: 50 (statistical significance)")
    print("="*80 + "\n")
    
    for num_drones in drone_counts:
        print(f"Testing with {num_drones:,} drones...", end=" ", flush=True)
        
        # Create environment and swarm
        env = Environment3D()
        original_num_boids = config.num_boids
        config.num_boids = num_drones
        
        swarm = SwarmManager3D(env)
        
        # Warm-up (JIT compilation, cache stabilization)
        for _ in range(10):
            swarm.update()
        
        # ─── Benchmark NAIVE ───────────────────────────────────────────────
        swarm.set_method('naive')
        naive_times = []
        for _ in range(50):
            t_start = time.perf_counter()
            swarm.update()
            t_end = time.perf_counter()
            naive_times.append((t_end - t_start) * 1000)  # ms
        
        naive_mean = np.mean(naive_times)
        naive_std = np.std(naive_times)
        
        # ─── Benchmark OCTREE ──────────────────────────────────────────────
        swarm.set_method('octree')
        octree_times = []
        for _ in range(50):
            t_start = time.perf_counter()
            swarm.update()
            t_end = time.perf_counter()
            octree_times.append((t_end - t_start) * 1000)  # ms
        
        octree_mean = np.mean(octree_times)
        octree_std = np.std(octree_times)
        
        # Calculate speedup
        speedup = naive_mean / octree_mean
        
        results.append({
            'drone_count': num_drones,
            'naive_mean_ms': naive_mean,
            'naive_std_ms': naive_std,
            'octree_mean_ms': octree_mean,
            'octree_std_ms': octree_std,
            'speedup': speedup,
            'winner': 'OCTREE' if speedup > 1.0 else 'NAIVE'
        })
        
        # Print progress
        print(f"Naive: {naive_mean:.2f}±{naive_std:.2f}ms | "
              f"Octree: {octree_mean:.2f}±{octree_std:.2f}ms | "
              f"Speedup: {speedup:.2f}x")
        
        # Restore
        config.num_boids = original_num_boids
    
    return results


def print_results_table(results):
    """Pretty-print results as ASCII table."""
    print("\n" + "="*100)
    print("  DETAILED RESULTS TABLE")
    print("="*100)
    print(f"{'Drones':>12} | {'Naive (ms)':>18} | {'Octree (ms)':>18} | {'Speedup':>10} | {'Winner':>8}")
    print("-"*100)
    
    for row in results:
        print(f"{row['drone_count']:>12,} | "
              f"{row['naive_mean_ms']:>8.2f}±{row['naive_std_ms']:<7.2f} | "
              f"{row['octree_mean_ms']:>8.2f}±{row['octree_std_ms']:<7.2f} | "
              f"{row['speedup']:>9.2f}x | "
              f"{row['winner']:>8}")
    
    print("="*100 + "\n")


def export_csv(results, filename="neighbor_algo_benchmark.csv"):
    """Export results to CSV for graphing."""
    with open(filename, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"[EXPORT] Results saved to {filename}")


def main():
    # Test scales: 100 to 2000 drones (exponential)
    drone_counts = [100, 250, 500, 1000, 1500, 2000]
    
    results = run_benchmark(drone_counts)
    print_results_table(results)
    export_csv(results)
    
    # Summary
    print("\n" + "="*80)
    print("  SUMMARY FOR PROFESSOR PRESENTATION")
    print("="*80)
    
    # Find crossover point (if any)
    octree_wins = [r for r in results if r['speedup'] > 1.0]
    naive_wins = [r for r in results if r['speedup'] <= 1.0]
    
    if octree_wins:
        print(f"\n✓ Octree wins in {len(octree_wins)}/{len(results)} cases")
        best_speedup = max(octree_wins, key=lambda x: x['speedup'])
        print(f"  Best speedup: {best_speedup['speedup']:.2f}x at {best_speedup['drone_count']:,} drones")
    
    if naive_wins:
        print(f"\n✓ Naive wins in {len(naive_wins)}/{len(results)} cases")
        print(f"  Efficient for small swarms where overhead matters")
    
    print("\n" + "="*80 + "\n")


if __name__ == "__main__":
    main()
