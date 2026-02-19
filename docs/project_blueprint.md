# Project Blueprint: 100 Drone Swarm Simulation

This document serves as the master guide for the team. We will execute this project in **two distinct phases**:
1.  **Phase 1:** Build a robust, fully functional **2D Simulation** to validate our algorithms and architecture.
2.  **Phase 2:** Upgrade the visualization and physics to **3D** once the core logic is proven.

---

## Phase 1: The 2D Foundation
**Goal:** demonstrate emergent behavior of 100 autonomous agents in a 2D plane with collision avoidance and task allocation.

### Team Roles & Responsibilities (4 Members)

| Role | Focus Area | Key Responsibilities (Phase 1) |
| :--- | :--- | :--- |
| **Member A** | **Physics & Engine** | - 2D Grid/Map setup <br> - Boundary handling (walls) <br> - Main Simulation Loop (Time steps) |
| **Member B** | **Swarm Logic** | - Drone Class (x, y, vx, vy) <br> - Flocking Rules (Sep, Ali, Coh) <br> - Steering behaviors |
| **Member C** | **Visualization** | - **Pygame/Matplotlib** rendering <br> - UI Dashboard (Sliders, Buttons) <br> - Visual Debugging (Force vectors) |
| **Member D** | **Optimization** | - **Quadtree** spatial partitioning <br> - Neighbor detection efficiency <br> - Parallel processing (Multiprocessing) |

---

### Step-by-Step Implementation Guide (2D)

#### Milestone 1: Environment & Basics
*   **[All] Setup:** Initialize Git repo. Install `pygame`, `numpy`, `matplotlib`.
*   **[Member A] The Arena:** Create a fixed 2D window (e.g., 800x600). Add static rectangular obstacles.
*   **[Member B] The Drone:** Create `Drone` class. Implement basic movement: `pos += vel * dt`. Handle distinct IDs.
*   **[Member C] The View:** Draw 100 circles (drones) and rects (obstacles). Add a "Pause/Play" button.
*   **[Member D] The Grid:** Implement a naive $O(N^2)$ neighbor check to start. Measure FPS as a baseline.

#### Milestone 2: Intelligent Swarm
*   **[Member B] Reynolds' Rules:** Implement the 3 core behaviors:
    1.  **Separation:** Steer away if `dist < r_separation`.
    2.  **Alignment:** Avg velocity of neighbors.
    3.  **Cohesion:** Steer towards avg position of neighbors.
*   **[Member A] Collisions:** Simple reflection: if hit wall, `vel *= -1`. If hit obstacle, slide or bounce.
*   **[Member D] Optimization:** Replace naive check with a **Quadtree** or **Spatial Hash Grid**. Goal: Maintain 60 FPS with 200 agents.
*   **[Member C] Interactive UI:** Add sliders to tweak weights: `Separation Weight`, `Cohesion Weight`, etc. real-time.

#### Milestone 3: Tasks & Parallelism
*   **[Member A] Dynamic World:** Make some obstacles move.
*   **[All] Logic Check:** Ensure sequential implementation works perfectly.
*   **[Member D] Parallelization:** Use Python `multiprocessing` to update drone states on multiple cores. Compare performance.
*   **[Member C] Reporting:** Integrate real-time graphs (e.g., "Average Velocity", "Clumper Factor").

---

## Phase 2: The 3D Upgrade
**Goal:** Port the proven 2D logic into a 3D environment for the final project showcase.
*   **Trigger:** Start this ONLY after Phase 1 is bug-free.

### Migration Steps
1.  **Visualization Upgrade (Member C):** Switch from Pygame to **Ursina Engine**.
    *   *Task:* Replace 2D circles with 3D Spheres or Drone Models.
    *   *Task:* Add Camera controls (Orbit, Zoom).
2.  **Physics Expansion (Member A & B):**
    *   *Task:* Update positions to `(x, y, z)`.
    *   *Task:* Update distance checks to Spherical ($d = \sqrt{x^2+y^2+z^2}$).
    *   *Task:* Boundary becomes a Cube/Box.
3.  **Optimization Upgrade (Member D):**
    *   *Task:* Convert **Quadtree** (2D) to **Octree** (3D).
    *   *Task:* Ensure parallel processing handles the 3rd dimension data.

---

## Resources & Research
The following resources are critical for understanding the "Boids" algorithm and existing Python implementations.

### Key Research Papers (2024-2026 Context)
*   **"Scalable Decentralized Control for Large-Scale Swarms"**: Minimizing communication bandwidth.
*   **"Reinforcement Learning for Collision Avoidance"**: Critical for dense 100+ agent swarms.
*   **"Fault-Tolerant Flocking"**: Robustness against drone failure.

### GitHub Code References
*   **[vmodel (Assessment & Stats)](https://github.com/lis-epfl/vmodel)**: Good for understanding statistical data generation.
*   **[MAVSDK Drone Show](https://github.com/alireza787b/mavsdk_drone_show)**: excellent reference for high-fidelity 3D simulation.
*   **[Drone Swarm Simulation](https://github.com/jeanjerome/drone-swarms)**: Simple Python + Matplotlib implementation (Good starting point for Phase 1).
*   **[Ursina Engine Documentation](https://www.ursinaengine.org/)**: Primary reference for Phase 2 visualization.

---

## Deliverables Checklist
- [ ] Source Code (GitHub Repo)
- [ ] 2D Demo Video (Milestone 2)
- [ ] 3D Final Demo Video (Phase 2)
- [ ] Performance Report (Sequential vs Parallel FPS)
- [ ] Project Report Document (LaTeX/PDF)
