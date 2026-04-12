"""
benchmark_final.py - Final benchmark for Milestone 2
Only uses working implementations
"""

import time
from environment import Environment
from swarm import SwarmManager
from swarm_vectorized import SwarmManagerVectorized
from swarm_parallel import SwarmManagerParallel

def benchmark(swarm_class, name, iterations=300, **kwargs):
    """Run benchmark and return time"""
    env = Environment("config.yaml")
    
    if 'num_processes' in kwargs:
        swarm = swarm_class(env, num_processes=kwargs['num_processes'])
    else:
        swarm = swarm_class(env)
    
    # Warmup
    for _ in range(50):
        swarm.update()
    
    start = time.perf_counter()
    for _ in range(iterations):
        swarm.update()
    end = time.perf_counter()
    
    return end - start

if __name__ == "__main__":
    print("=" * 60)
    print("MILESTONE 2 - FINAL BENCHMARK RESULTS")
    print("=" * 60)
    
    iterations = 300
    
    # Original
    print("\nRunning Original...")
    orig = benchmark(SwarmManager, "Original", iterations)
    print(f"  Original: {orig:.2f}s")
    
    # Vectorized
    print("\nRunning Vectorized...")
    vec = benchmark(SwarmManagerVectorized, "Vectorized", iterations)
    vec_speedup = orig / vec
    print(f"  Vectorized: {vec:.2f}s → {vec_speedup:.2f}x")
    
    # Parallel (2 cores)
    print("\nRunning Parallel (2 cores)...")
    par2 = benchmark(SwarmManagerParallel, "Parallel", iterations, num_processes=2)
    par2_speedup = orig / par2
    print(f"  Parallel (2 cores): {par2:.2f}s → {par2_speedup:.2f}x")
    
    # Parallel (4 cores)
    print("\nRunning Parallel (4 cores)...")
    par4 = benchmark(SwarmManagerParallel, "Parallel", iterations, num_processes=4)
    par4_speedup = orig / par4
    print(f"  Parallel (4 cores): {par4:.2f}s → {par4_speedup:.2f}x")
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Original:          {orig:.2f}s (baseline)")
    print(f"Vectorized:        {vec:.2f}s → {vec_speedup:.2f}x faster")
    print(f"Parallel (2 cores): {par2:.2f}s → {par2_speedup:.2f}x faster")
    print(f"Parallel (4 cores): {par4:.2f}s → {par4_speedup:.2f}x faster")