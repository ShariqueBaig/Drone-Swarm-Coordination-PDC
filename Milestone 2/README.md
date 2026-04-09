# Milestone 2: Decentralized Coordination Algorithms

## Expected Role and Milestone Goal
The objective of Milestone 2 is to implement distributed algorithms enabling collective behavior without central supervision. The swarm must dynamically assign and resolve missions through decentralized auctioning, reach consensus without a global server, and form complex macroscopic structures.

**Trigger to proceed to Milestone 3:** Emergent coordinated motion (flocking/formation) visible, and dynamic assignment/completion of environmental waypoints validated.

---

## What Sharique Has Done (Swarm Behavioral Logic)
Sharique is responsible for the core **Swarm Behavioral Logic**. His completed tasks for this milestone are fully integrated into `src/swarm.py` and substantially evolve the complexity of the swarm engine:

- **B2.1 (Reynolds Flocking Integration):** Seamlessly bridged the prior separation, alignment, and cohesion steering logic into a unified framework alongside the new task and formation weights.
- **B2.2 (Decentralized Task Allocation):** Implemented an auction protocol. The environment generates random waypoint tasks. If unassigned, each drone evaluates its relative proximity to the tasks and internally generates a spatial "bid" corresponding to its distance to the closest available task. 
- **B2.3 (Consensus Mechanisms):** Constructed conflict resolution for structural bidding. If numerous drones target the same node, they evaluate competing bids without a master server. The highest priority (closest distance, with ID resolving perfect ties) asserts dominance, safely forcing the losing drone to relinquish its assignment dynamically.
- **B2.4 (Local Communication Protocols):** Injected a `communication_radius` variable (`src/config.py`). Drones are strictly limited to broadcasting intents and bids *(assigned target location + priority)* to immediate neighbors inside this radius, proving pure localized network topology algorithms.
- **B2.5 (Formation Control):** Added `calculate_formation_steer()` utilizing virtual group centroid tracking. Dependent on the vector heading of the collective cluster, the swarm computes indexed spatial offsets (V-formations) dynamically resulting in macro emergent geometries navigating space.

---

## Project Structure (Milestone 2 Scope)

The codebase in this folder `src` directly builds on top of the environment initialized prior:

```text
Milestone 2/
├── README.md
└── src/
    ├── config.py          ← Sharique added M2 parameters (task_weight, communication_radius)
    ├── environment.py     ← Core physics
    ├── swarm.py           ← Sharique (B2.1-B2.5): Auction, Consensus, and Formation Logic
    └── main.py            ← Shared integration entry point
```

## How To Run & Test Sharique's Task Assignments

To witness the decentralized logic (Auctioning, Communication, and Task consensus tracking):

1. **Navigate to the Milestone 2 folder:**
   ```bash
   cd "Milestone 2"
   ```
2. **Run the simulation backbone script:**
   ```bash
   python src/main.py
   ```

*(Note: While the headless physics/swarm engine handles the tasks flawlessly within the backend memory matrix, visual UI components for 'task waypoint zones' requested in C2.3 by the Visualization role can easily extract data from `SwarmManager.assigned_tasks` array when they integrate!)*
