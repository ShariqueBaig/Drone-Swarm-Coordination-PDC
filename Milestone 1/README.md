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

## What Suffiyan Has Done
Suffiyan is responsible for the **Physics & Environment Engine**. His completed tasks for this milestone are fully integrated and provide the physical foundation for the swarm:

- **A1.1 & A1.3 (Environment & Loop):** Established a 1000x1000 unit continuous plane with a fixed simulation time-step (`dt = 0.02s`).
- **A1.2 (Static Obstacles):** Implemented a system for circular obstacles that drones can sense and avoid. These are stored as a configurable list of `(x, y, radius)`.
- **A1.4 (Boundary Physics):** Developed robust boundary handling, including "hard-wall" repulsion forces that push drones away from edges and "wrap-around" logic.
- **A1.5 (Configuration Management):** Created `config.yaml` to centralize all simulation parameters (world size, obstacles, drone count, seed), ensuring the environment is easily tunable without touching code.

## What Ashhal Has Done
Ashhal is responsible for **Optimization & Parallelism**. His contributions ensure the simulation remains high-performance even as the swarm scales:

- **D1.1 - D1.3 (Spatial Partitioning):** Implemented advanced spatial indexing algorithms including **Grid Hash** (`spatial_grid.py`) and **Quadtree** (`quadtree.py`) to reduce neighbor-sensing complexity from $O(N^2)$ to $O(N)$.
- **D1.4 (Performance Logging):** Developed a comprehensive benchmarking suite (`performance_logger.py`) that tracks real-time FPS, CPU, and memory metrics, outputting data for analysis.
- **D1.5 (Vectorization Baseline):** Ensured all optimization modules utilize NumPy broadcasting to maximize data parallelism.

## Project Structure & Team Contracts
The codebase has been refactored to meet the modular architecture agreed upon by the team (Usman's contract):

```text
pdc_project/
├── src/
│   ├── config.py          ← Shared constants
│   ├── config.yaml        ← Environmental parameters (A1.5)
│   ├── environment.py     ← Suffiyan (A1.1-A1.4): World physics
│   ├── swarm.py           ← Sharique (B1.1-B1.5): SwarmManager logic
│   ├── spatial_grid.py    ← Ashhal (D1.2): Grid Hash indexing
│   ├── quadtree.py        ← Ashhal (D1.3): Quadtree indexing 
│   ├── performance_logger.py ← Ashhal (D1.4): Metrics telemetry
│   ├── visualizer.py      ← Usman (C1.1-C1.5): Rendering loop
│   └── main.py            ← Shared integration entry point
```

- **`src/environment.py`:** Handles world physics, boundary resolution, and obstacle data.
- **`src/swarm.py`:** Contains the main SwarmManager and behavioral logic.
- **`src/spatial_grid.py` / `src/quadtree.py`:** Optimization layers for fast neighbor lookups.
- **`src/main.py`:** The integration layer that wires the environment, swarm, and visualizer together. By default, it uses Ashhal's **Optimized Swarm Manager** for superior performance.

## Performance Telemetry
Ashhal's optimization layer includes a real-time performance logger.
- **`src/optimized_benchmark.csv`**: Automatically logs FPS and system usage.
- **`src/view_logs.py`**: Run this to generate visual performance graphs of your swarm!
  ```bash
  python src/view_logs.py
  ```

## Running the Simulation
To verify the milestone progress thus far, run the shared entry point from the root of this folder:
```bash
python src/main.py
```
