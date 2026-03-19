# Milestone 1: Environmental Modeling and Drone Behavior Foundation

## Expected Role and Milestone Goal
The objective of Milestone 1 is to establish the simulated environment and foundational drone control logic in an interactive 2D simulation. 

**Trigger to proceed to Milestone 2:** 100 drones spawn randomly, move, and sense neighbors without collisions.

## What Sharique Has Done
Sharique is responsible for the core **Swarm Behavioral Logic**. His completed tasks for this milestone are fully integrated into the `src` module and include:

- **B1.1 & B1.2 (Drone Representation and Kinematics):** Drones are vectorized utilizing standard properties (`pos`, `vel`, `heading`). Positional updates integrate correctly using a strict simulated time-step (`dt = 0.02s`). To counteract the smaller time-step, engine caps were upscaled significantly (`max_speed = 250`, `max_force = 5.0`) to provide natural visual movement. Drones are given explicit `id` arrays for internal tracking control.
- **B1.3 (Random Initialization):** Drones scatter randomly within the simulation boundaries. The system initializes off a configurable seed (`seed = 42`), establishing perfectly reproducible conditions for visual and algorithm debugging.
- **B1.4 (Basic Drone Sensing):** Drones observe immediate neighbors through vectorized distance-matrix calculations across their defined perception radii (`R = 50`).
- **B1.5 (Reactive Avoidance):** Beyond general awareness, the core flocking module establishes an urgent `safety_distance` (20 units). By strongly repulsing neighbors within this radius, the drones naturally organize into a non-colliding **emergent lattice (crystalline structure)**, definitively fulfilling the true milestone physics requirements!

*(Note: The `src/` code structure currently emphasizes Sharique's completed Behavioral Logic implementation. Code additions mapping to tasks by other teammates—Suffiyan, Usman, Asshal—will be integrated directly into this module soon!)*

## Project Structure & Team Contracts
The codebase has been refactored to meet the modular architecture agreed upon by the team (Usman's contract):

```text
pdc_project/
├── config.py          ← Shared constants
├── environment.py     ← Suffiyan (A1.1-A1.5)
├── swarm.py           ← Sharique (B1.1-B1.5) [SwarmManager]
├── optimization.py    ← Ashhal (D1.1-D1.5)
├── visualizer.py      ← Usman (C1.1-C1.5)
└── main.py            ← Shared entry point
```

- **`swarm.py`:** Contains the physics and behavioral logic. It exposes drone positions and velocities as NumPy arrays.
- **`visualizer.py`:** Standalone rendering engine that consumes swarm data for visualization.
- **`main.py`:** The integration layer that wires the environment, swarm, and visualizer together.

## Running the Simulation
To verify the milestone progress thus far, run the shared entry point from the root of this folder:
```bash
python src/main.py
```
