"""
benchmark_parallel_vs_sequential.py - Compare sequential vs parallel performance
"""

import time
import numpy as np
from environment import Environment
from swarm import SwarmManager
from swarm_parallel import SwarmManagerParallel

def benchmark_original(iterations=500):
    """Benchmark original SwarmManager"""
    env = Environment("config.yaml")
    swarm = SwarmManager(env)
    
    start = time.perf_counter()
    for i in range(iterations):
        swarm.update()
    end = time.perf_counter()
    
    return end - start

def benchmark_parallel(iterations=500, num_processes=2):
    """Benchmark parallel SwarmManager"""
    env = Environment("config.yaml")
    swarm = SwarmManagerParallel(env, num_processes=num_processes)
    
    start = time.perf_counter()
    for i in range(iterations):
        swarm.update()
    end = time.perf_counter()
    
    return end - start

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL VS SEQUENTIAL BENCHMARK - Milestone 2")
    print("=" * 60)
    
    iterations = 300  # Fewer iterations for faster testing
    
    # Original baseline
    print(f"\nRunning ORIGINAL SwarmManager ({iterations} updates)...")
    orig_time = benchmark_original(iterations)
    print(f"  Original: {orig_time:.2f} seconds")
    
    # Test different process counts
    results = []
    for cores in [1, 2, 4]:
        print(f"\nRunning PARALLEL with {cores} core(s)...")
        try:
            par_time = benchmark_parallel(iterations, cores)
            speedup = orig_time / par_time
            print(f"  Parallel ({cores} cores): {par_time:.2f} seconds")
            print(f"  Speedup: {speedup:.2f}x")
            results.append((cores, par_time, speedup))
        except Exception as e:
            print(f"  Error: {e}")
            results.append((cores, 0, 0))
    
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(f"Original: {orig_time:.2f}s (baseline)")
    for cores, par_time, speedup in results:
        if par_time > 0:
            print(f"{cores} core(s): {par_time:.2f}s → {speedup:.2f}x speedup")
    
    # Save results
    with open('parallel_benchmark_results.txt', 'w') as f:
        f.write("M2 Parallel Processing Benchmark Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Original time: {orig_time:.2f}s\n\n")
        for cores, par_time, speedup in results:
            if par_time > 0:
                f.write(f"{cores} core(s): {par_time:.2f}s -> {speedup:.2f}x speedup\n")
    
    print("\n✅ Results saved to parallel_benchmark_results.txt")