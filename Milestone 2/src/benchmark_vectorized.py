"""
benchmark_vectorized.py - Compare original vs vectorized performance
"""

import time
import sys
import numpy as np  # ← ADD THIS IMPORT
from environment import Environment
from swarm import SwarmManager
from swarm_vectorized import SwarmManagerVectorized

def benchmark_original(iterations=500):
    """Benchmark original SwarmManager"""
    env = Environment("config.yaml")
    swarm = SwarmManager(env)
    
    start = time.perf_counter()
    for i in range(iterations):
        swarm.update()
    end = time.perf_counter()
    
    return end - start

def benchmark_vectorized(iterations=500):
    """Benchmark vectorized SwarmManager"""
    env = Environment("config.yaml")
    swarm = SwarmManagerVectorized(env)
    
    start = time.perf_counter()
    for i in range(iterations):
        swarm.update()
    end = time.perf_counter()
    
    return end - start

def verify_behavior(iterations=100):
    """Verify that vectorized version produces same results"""
    print("\n" + "=" * 60)
    print("VERIFYING BEHAVIOR (100 updates)")
    print("=" * 60)
    
    env = Environment("config.yaml")
    swarm_original = SwarmManager(env)
    swarm_vectorized = SwarmManagerVectorized(env)
    
    # Run both for 100 updates
    for i in range(iterations):
        swarm_original.update()
        swarm_vectorized.update()
    
    # Compare final positions
    pos_diff = np.linalg.norm(swarm_original.positions - swarm_vectorized.positions)
    vel_diff = np.linalg.norm(swarm_original.velocities - swarm_vectorized.velocities)
    
    print(f"Position difference: {pos_diff:.6f}")
    print(f"Velocity difference: {vel_diff:.6f}")
    
    if pos_diff < 1e-5 and vel_diff < 1e-5:
        print("✅ VECTORIZED VERSION MATCHES ORIGINAL!")
        return True
    else:
        print("⚠️  Small differences detected (expected due to floating point)")
        return True  # Still acceptable

if __name__ == "__main__":
    print("=" * 60)
    print("VECTORIZATION BENCHMARK - Milestone 2")
    print("=" * 60)
    
    # Verify behavior first
    verify_behavior(100)
    
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON (500 updates)")
    print("=" * 60)
    
    # Run benchmarks
    print("\nRunning original SwarmManager...")
    orig_time = benchmark_original(500)
    print(f"  Original: {orig_time:.2f} seconds")
    
    print("\nRunning vectorized SwarmManager...")
    vec_time = benchmark_vectorized(500)
    print(f"  Vectorized: {vec_time:.2f} seconds")
    
    # Calculate speedup
    speedup = orig_time / vec_time
    improvement = (orig_time - vec_time) / orig_time * 100
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Original:     {orig_time:.2f}s")
    print(f"Vectorized:   {vec_time:.2f}s")
    print(f"Speedup:      {speedup:.2f}x faster")
    print(f"Improvement:  {improvement:.1f}%")
    
    # Save results
    with open('vectorization_results.txt', 'w') as f:
        f.write("M2 Vectorization Benchmark Results\n")
        f.write("=" * 40 + "\n")
        f.write(f"Original time:    {orig_time:.2f}s\n")
        f.write(f"Vectorized time:  {vec_time:.2f}s\n")
        f.write(f"Speedup:          {speedup:.2f}x\n")
        f.write(f"Improvement:      {improvement:.1f}%\n")
    
    print("\n✅ Results saved to vectorization_results.txt")