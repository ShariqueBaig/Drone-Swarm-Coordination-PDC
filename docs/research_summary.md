# Research Summary: 100 Drone Coordination Simulation

## 1. State of the Art: Decentralized Drone Swarm Coordination (2024-2026)

Recent research emphasizes moving away from centralized control to fully decentralized systems where each drone makes independent decisions based on local sensing.

### Key Trends & Algorithms:
*   **Bio-Inspired Swarm Intelligence:** Algorithms like **Boids (Reynolds)** remain foundational. Recent advancements integrate **Particle Swarm Optimization (PSO)** and **Ant Colony Optimization (ACO)** for dynamic path planning.
*   **Deep Reinforcement Learning (DRL):** Using Multi-Agent Reinforcement Learning (MARL) to train drones to "learn" coordination strategies (e.g., collision avoidance, formation flying) rather than hard-coding rules.
*   **Hybrid Architectures:** Combining rule-based reactive behaviors (fast, safe) with AI-driven consensus (smart, strategic).
*   **Communication Protocols:** Research focuses on "Flying Ad-Hoc Networks" (FANETs) where drones act as network nodes, sharing state data (position, velocity) with neighbors only.

### Relevant Research Papers/Topics (Feb 2026 Context):
*   **"Scalable Decentralized Control for Large-Scale Swarms"**: Focuses on minimizing communication bandwidth while maintaining formation.
*   **"Reinforcement Learning for Collision Avoidance in Dense Swarms"**: key for 100+ agents.
*   **"Fault-Tolerant Flocking"**: How swarms recover when individual drones fail or are removed (robustness).

## 2. GitHub Repositories & Code Resources

The following repositories are relevant starting points for "100 Drone" simulations in Python:

*   **[vmodel](https://github.com/lis-epfl/vmodel)**: Vision-based swarm simulation. Designed for generating statistical data for large groups (1000+ agents).
*   **[MAVSDK Drone Show](https://github.com/alireza787b/mavsdk_drone_show)**: High-fidelity simulation for drone shows. Includes "100-Drone SITL Test" cases.
*   **[Drone Swarm Simulation](https://github.com/jeanjerome/drone-swarms)**: Python + Matplotlib/Tkinter. Implements consensus and formation control.
*   **[DSSE (Drone Swarm Search Environment)](https://pypi.org/project/DSSE/)**: Focuses on maritime search and rescue with swarms.
*   **[UAV-Swarm-Simulator](https://github.com/matteoprata/UAV-Swarm-Simulator)**: Framework for defining behaviors and base stations.

## 3. Sequential vs. Parallel Implementation

Your project document asks for both. Here is the technical breakdown:

### Sequential Code (Baseline)
*   **Logic:** The program iterates through the list of drones one by one.
    *   `for drone in drones: update_position()`
    *   `for drone in drones: check_neighbors()`
*   **Complexity:** Naive neighbor checking is **O(N²)**. For 100 drones, this is 10,000 checks per frame. Feasible in Python, but scales poorly.
*   **Pros:** Easier to debug, no race conditions, deterministic.
*   **Cons:** Slows down significantly as N increases > 200.

### Parallel Code (Advanced)
*   **Logic:** Distributes the workload across multiple CPU cores.
*   **Approaches:**
    1.  **Multiprocessing (CPU):** Split the list of drones into chunks. Core 1 updates drones 0-25, Core 2 updates 26-50, etc. Requires shared memory or message passing for position data.
    2.  **Vectorization (NumPy/GPU):** Instead of Python loops, use matrix operations to update all state vectors `[x, y, vx, vy]` simultaneously. This is "data parallelism".
*   **Pros:** Scales to thousands of agents.
*   **Cons:** Harder to implement (synchronization issues), overhead of spawning processes.

## 4. Upgrading to 3D Simulation (New Requirement)

Moving from 2D to 3D increases complexity in visualization and math, but is feasible for 100 agents.

### Visualization Libraries
*   **[Ursina Engine](https://www.ursinaengine.org/):** (Recommended)
    *   **Pros:** Easiest to use, "game-like" visuals, built-in 3D camera/controls. Pythonic API.
    *   **Cons:** Might hit performance limits if scaling > 500 agents without optimization.
*   **[VisPy](https://vispy.org/):**
    *   **Pros:** High-performance (OpenGL), handles 1000+ points easily.
    *   **Cons:** Steeper learning curve, less "out of the box" game features.
*   **Matplotlib 3D:**
    *   **Pros:** Standard library, easy to debug.
    *   **Cons:** **Not recommended** for real-time swarms. Slow frame rates (>20 FPS is hard) and lacks depth perception cues.

### Algorithm Adjustments
*   **Kinematics:** State vector becomes `[x, y, z, vx, vy, vz]`.
*   **Neighbor Search:** Distance formula becomes $d = \sqrt{\Delta x^2 + \Delta y^2 + \Delta z^2}$.
    *   *Optimization:* 3D Grid (Voxel) partitioning or Octrees are needed instead of Quadtrees.
*   **Flocking Rules:**
    *   *Separation:* Must avoid collisions in 3D space (up/down as well as left/right).
    *   *Alignment/Cohesion:* Vectors are now 3D.

### Recommendation
*   Start with **Sequential** + **Spatial Partitioning** (Grid/Quadtree). This reduces O(N²) to O(N) by only checking nearby cells.
*   Implement **Parallel** execution (via Python `multiprocessing` or `Ray`) as an optimization milestone.
*   **Use Ursina** for the 3D visualization to ensure a "premium" look and easy 3D navigation.
