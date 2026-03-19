# main.py is where everything connects:
try:
    from visualizer import run_viz
    from swarm import SwarmManager
    from environment import Environment
except ImportError as e:
    print(f"Import Error in Modular Structure: {e}")
    import sys
    sys.exit(1)

# Initialize Environment (Suffiyan's A1.1-A1.5)
env = Environment("config.yaml")

# Initialize Swarm (Sharique's B1.1-B1.5)
swarm_manager = SwarmManager(env)

run_viz(swarm_mgr=swarm_manager, env=env)
