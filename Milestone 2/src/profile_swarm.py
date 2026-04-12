#!/usr/bin/env python
"""
Bottleneck Analysis for Milestone 2
Run: python profile_swarm.py
"""
import cProfile
import pstats
import io
import sys  # ← ADD THIS
from environment import Environment
from swarm import SwarmManager

def run_simulation(updates=500):
    env = Environment("config.yaml")
    swarm = SwarmManager(env)
    for i in range(updates):
        swarm.update()
        if i % 100 == 0:
            print(f"  Progress: {i}/{updates}")

if __name__ == "__main__":
    print("=" * 60)
    print("M2 BOTTLENECK ANALYSIS - Run this FIRST")
    print("=" * 60)
    
    profiler = cProfile.Profile()
    profiler.enable()
    run_simulation(500)
    profiler.disable()
    
    print("\n" + "=" * 60)
    print("TOP 20 MOST EXPENSIVE FUNCTIONS")
    print("=" * 60)
    
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    stats.dump_stats('swarm_profile.prof')
    print("\n✅ Profile saved to 'swarm_profile.prof'")
    print("\n🔍 Look for:")
    print("   - auction_tasks() - communication overhead")
    print("   - calculate_formation_steer() - sequential loop")
    print("   - np.linalg.norm calls - repeated calculations")