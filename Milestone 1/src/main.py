# main.py is where everything connects:
try:
    from visualizer import main as run_viz
    from swarm import SwarmManager
except ImportError as e:
    print(f"Import Error in Modular Structure: {e}")
    import sys
    sys.exit(1)

# Sharique, Suffiyan, Ashhal initialize their objects here
swarm_manager = SwarmManager()
obstacles = [] # List[pygame.Rect] to be explicitly managed by Suffiyan (Environment layer)

run_viz(swarm_mgr=swarm_manager, environment_obstacles=obstacles)
