try:
    from environment import Environment
    from swarm_optimized import SwarmManagerOptimized as SwarmManager
    from visualizer import run_viz
except ImportError as e:
    print(f"Import Error in Modular Structure: {e}")
    import sys
    sys.exit(1)

# Initialize Environment (Suffiyan's A1.1-A1.5)
env = Environment("config.yaml")

# Initialize Optimized Swarm (Sharique's logic + Ashhal's D1.1-D1.5 optimizations)
swarm_manager = SwarmManager(env)

# Run the simulation
run_viz(swarm_mgr=swarm_manager, env=env)
