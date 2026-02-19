# Team Execution Plan: 100 Drone Coordination Simulation

## Project Feasibility Check
**Verdict: FEASIBLE & WELL-SCOPED**
The project guidelines are technically sound. 100 agents in a 2D discrete simulation is a classic computer science problem (Boids).
*   **Computational Load:** Manageable in Python if efficient algorithms (spatial partitioning) are used.
*   **Complexity:** Decentralized logic is challenging but modular.

---

## Task Distribution Strategy (4 Members)
To ensure balanced workloads and parallel progress, we've divided the team into 4 specialized roles:

1.  **Member A: Physics & Environment Engine**
    *   Focus: 3D world boundaries, collision physics (walls/obstacles), and the main simulation loop.
2.  **Member B: Swarm Behavioral Logic**
    *   Focus: Core flocking algorithms (Separation, Alignment, Cohesion) and steering vector integration.
3.  **Member C: Visualization & UI/UX**
    *   Focus: 3D rendering (Ursina/VisPy), camera controls, and the interactive dashboard (UI).
4.  **Member D: Coordination & Systems Optimization**
    *   Focus: Spatial partitioning (Octree/Voxel Grid), decentralized task allocation, and parallelization.

---

## Detailed Task Breakdown

### Milestone 1: Environment & Foundation

| Task ID | Task Name | Description | Assignee |
| :--- | :--- | :--- | :--- |
| **M1.1** | Setup Repo structure | Create Git repo, folders, and `requirements.txt` (add `ursina`, `numpy`). | All |
| **M1.2** | Build Simulation Loop | Create main loop `Update() -> Render()` with fixed time step. | Member A |
| **M1.3** | Implement 3D Space (Box) | Define 3D environment boundaries (x, y, z) and static obstacle cubes. | Member A |
| **M1.4** | Create Drone Class (3D) | Define class with properties: `pos(x,y,z)`, `vel(vx,vy,vz)`, `id`. | Member B |
| **M1.5** | Implement 3D Kinematics | Write `update_position()` function based on 3D velocity vectors. | Member B |
| **M1.6** | Multi-Drone Rendering | Use **Ursina** to efficiently draw 100 spheres/cubes. | Member C |
| **M1.7** | Interactive Camera | Implement Orbit, Pan, and Zoom controls for the 3D view. | Member C |
| **M1.8** | Neighbor Detection | Implement function to find drones within radius `R` (sphere check). | Member D |
| **M1.9** | Baseline Performance | Track FPS with 100 drones in sequential O(N²) mode. | Member D |

### Milestone 2: Decentralized Algorithms

| Task ID | Task Name | Description | Assignee |
| :--- | :--- | :--- | :--- |
| **M2.1** | 3D Collision Physics | Detect when drones hit walls/obstacles and resolve bounces/stops. | Member A |
| **M2.2** | Separation Force (3D) | Code steering force to avoid crowding neighbors. | Member B |
| **M2.3** | Alignment & Cohesion | Code steering forces for 3D group cohesion and alignment. | Member B |
| **M2.4** | UI Dashboard | Build an overlay with sliders for Flocking Weights (Sep, Ali, Coh). | Member C |
| **M2.5** | Visual Debuggers | Draw lines to neighbors or show velocity vectors in 3D. | Member C |
| **M2.6** | Spatial Partitioning (3D) | Optimize neighbor detection using an **Octree** or **3D Grid**. | Member D |
| **M2.7** | Task Allocation (Base) | Implement state machine: `Searching` -> `Engaging` -> `Returning`. | Member D |

### Milestone 3: Tasks & Validation

| Task ID | Task Name | Description | Assignee |
| :--- | :--- | :--- | :--- |
| **M3.1** | Dynamic Obstacles | Add moving spheres/cubes for drones to avoid in 3D. | Member A |
| **M3.2** | Advanced Rule System | Add "Target Seeking" and "Obstacle Avoidance" steering. | Member B |
| **M3.3** | UI Metrics Display | Add real-time graphs for Collision Count and Area Coverage. | Member C |
| **M3.4** | Parallel Update System | Convert drone updates to run on a `multiprocessing` pool. | Member D |
| **M3.5** | Mission Execution | Setup scenario: "100 Drones covering a 3D volume". | All |
| **M3.6** | Final Performance Log | Comparison report: Sequential vs Parallel on 1, 2, 4 cores. | Member D |

---

## Sequential vs Parallel Code Tasks
*   **Sequential Version:** Standard Python loop. (Deliverable for M1/M2)
*   **Parallel Version:** Use `multiprocessing.Pool` to update drone states.
    *   *Challenge:* The primary overhead in Python is data pickling between processes.
    *   *Task:* Role C should investigate `shared_memory` or `Ray` for efficient state sharing.
