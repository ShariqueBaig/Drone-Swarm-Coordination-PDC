# PDC Drone Swarm Coordination Simulation (2D)

This project simulates a fully decentralized swarm of 100 autonomous drones coordinating to achieve complex tasks without global supervision. Developed for the Parallel and Distributed Computing (PDC) course, it demonstrates emergent behavior, collision avoidance, distributed task allocation, and multi-threading performance optimizations.

## 🚀 Key Features

### 1. Decentralized Coordination & Flocking
Drones operate using Reynolds-style "Boids" rules (Separation, Alignment, Cohesion). There is no central controller micromanaging their flight paths; instead, they compute localized steering forces based purely on their immediate neighbors (within a `perception_radius`).

### 2. Local Message Passing & Consensus
To demonstrate true decentralization, global information is artificially restricted:
- **Limited Sensing**: When a mission is issued, only a small percentage of drones (those physically closest to the target) detect the task.
- **Gossip Protocol (Consensus)**: As "informed" drones fly past "uninformed" drones, they share the task coordinates via local message passing. 
- *Visual Indicator*: Drones flash a **bright yellow ring** the exact moment they receive a new task from a neighbor.

### 3. Emergent Mission Behaviors
The swarm can execute four complex missions through emergent teamwork:
- **Area Coverage**: Drones spread out using a greedy spatial search to map unvisited sectors. If a drone gets stuck, it uses a **Work-Stealing** queue to pull unvisited coordinates from other areas of the map.
- **Target Tracking**: The swarm autonomously tracks a moving target that dynamically bounces around the environment.
- **Formation Traversal**: Drones mathematically calculate their center of mass and average heading to assemble into a rigid, perfectly centered square grid. This grid dynamically resizes if drones are killed.
- **Object Transport / Recall**: Drones form sub-teams dynamically based on proximity to carry out fetch-and-retrieve tasks or gather at a rally point.

### 4. Dynamic Environment & Resilience
- **Real-Time Obstacle Avoidance**: Predictive collision detection and edge-based repulsion prevent drones from crashing into static walls and moving threats.
- **Fault Tolerance**: The swarm is resilient to sudden drone deaths. If faults are injected, surviving drones dynamically recalculate their formations and adapt task coverage without breaking the system.

### 5. PDC Performance Optimizations
- **Vectorization (SIMD)**: All physics math is heavily vectorized using NumPy, processing 100 drones simultaneously rather than using slow Python `for` loops.
- **Fork-Join Parallelism**: Steering calculations (forces, obstacle avoidance, boundary checks) are distributed across a `ThreadPoolExecutor`.
- **Spatial Partitioning**: Naive O(N²) neighbor checks can be hot-swapped to O(N log N) **KD-Trees** or **Grid Hashing** algorithms for massive speedups.
- **Decoupled Architecture**: The Pygame renderer runs independently from the physics thread, utilizing lock-free state snapshotting to maintain high FPS even under heavy computational load.

## 🎮 Controls

### Keyboard Hotkeys
| Key | Action |
|-----|--------|
| `SPACE` | Pause / Resume simulation |
| `R` | Reset swarm to Idle / Flocking |
| `F` | Inject faults (Instantly kills 20% of active drones) |
| `N` | Toggle **Neighbor Awareness** (draws lines between communicating drones) |
| `O` | Spawn a moving/dynamic obstacle |
| `H` | Toggle Area Coverage heatmap overlay |
| `1` | Switch neighbor search to Naive (Brute Force) |
| `2` | Switch neighbor search to KDTree (Quadtree) |
| `3` | Switch neighbor search to Grid Hashing |
| `ESC` | Quit |

### Mouse Controls
- **Left Click**: Select Mission in Fleet Command panel / Set global Waypoint.
- **Right Click**: Clear global Waypoint.
- **Shift + Left Click**: Place a static obstacle at mouse cursor.
- **Shift + Right Click**: Remove an obstacle at mouse cursor.

## 🛠️ Installation & Setup
1. Ensure Python 3.9+ is installed.
2. Install dependencies:
   ```bash
   pip install pygame numpy scipy pyyaml
   ```
3. Run the simulation:
   ```bash
   python simulation.py
   ```
