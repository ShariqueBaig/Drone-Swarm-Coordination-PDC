try:
    from environment import Environment
    from swarm_optimized import SwarmManagerOptimized as SwarmManager
    from visualizer import run_viz
    # NEW: Import your logger
    from performance_logger import PerformanceLogger
except ImportError as e:
    print(f"Import Error in Modular Structure: {e}")
    import sys
    sys.exit(1)

# Initialize Environment (Suffiyan's A1.1-A1.5)
env = Environment("config.yaml")

# Initialize Optimized Swarm (Sharique's logic + Ashhal's D1.1-D1.5 optimizations)
swarm_manager = SwarmManager(env)

# NEW: Initialize Logger
perf_logger = PerformanceLogger("swarm_benchmark.csv")

# Run the simulation
try:
    # Pass the logger into the visualizer
    run_viz(swarm_mgr=swarm_manager, env=env, logger=perf_logger)
finally:
    # CRITICAL: This ensures the final buffer is written to the file
    print("Saving performance logs...")
    perf_logger.close()