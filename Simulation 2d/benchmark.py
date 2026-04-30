
import time
import numpy as np
import config
from environment import Environment
from swarm import SwarmManager

def run_benchmark(num_frames=100, method="naive"):
    env = Environment()
    swarm = SwarmManager(env)
    swarm.set_method(method)
    
    print(f"--- Benchmarking M3 | Method: {method.upper()} | Drones: {config.num_boids} ---")
    
    start_time = time.time()
    for _ in range(num_frames):
        swarm.update()
    end_time = time.time()
    
    total_time = end_time - start_time
    tps = num_frames / total_time
    
    print(f"Total Time: {total_time:.4f}s")
    print(f"TPS (Ticks Per Second): {tps:.2f}")
    
    # Print Parallel Metrics
    swarm.metrics.print_report()
    swarm.metrics.export_csv("parallel_analysis.csv")
    return tps

if __name__ == "__main__":
    tps_naive = run_benchmark(200, "naive")
    tps_grid = run_benchmark(200, "grid")
    tps_octree = run_benchmark(200, "octree")
    
    print("\n" + "="*40)
    print("M3 PERFORMANCE SUMMARY")
    print("="*40)
    print(f"Naive (HPC Optimized): {tps_naive:.2f} TPS")
    print(f"Grid:                  {tps_grid:.2f} TPS")
    print(f"Octree:                {tps_octree:.2f} TPS")
    print("="*40)
