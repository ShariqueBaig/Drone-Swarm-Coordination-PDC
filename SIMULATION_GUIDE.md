# Drone Swarm Simulation Guide

This project includes two complementary simulations demonstrating different approaches to multi-agent coordination: **decentralized** and **centralized**.

---

## 1. Decentralized Simulation (`simulation.py`)

### Overview
The decentralized simulation demonstrates **autonomous swarm behavior** where 100 drones coordinate without any central authority. Each drone makes decisions based only on **local sensing and interaction** with nearby neighbors.

### Key Features

- **Local Sensing**: Each drone only detects neighbors within an `INTERACTION_RADIUS` of 50 units
- **No Global Knowledge**: Drones don't know positions of drones beyond their interaction radius
- **Scalable**: Can easily scale to thousands of drones without performance degradation (compared to centralized)
- **Resilient**: Swarm continues functioning even if individual drones fail

### Behavior Mechanisms

The simulation implements **Reynolds-style flocking** with three core forces:

#### 1. **Separation** (Collision Avoidance)
- Prevents drones from crowding each other
- Steers away from neighbors that are too close
- Weight: `SEPARATION_WEIGHT = 1.5`

#### 2. **Alignment** (Velocity Matching)
- Drones match the heading of nearby peers
- Encourages coordinated movement direction
- Weight: `ALIGNMENT_WEIGHT = 1.0`

#### 3. **Cohesion** (Attraction to Group Center)
- Drones steer toward the center of mass of their local group
- Keeps the swarm together
- Weight: `COHESION_WEIGHT = 1.0`

#### 4. **Obstacle Avoidance**
- Drones detect static obstacles (red circles) within a safety distance of `SAFETY_DISTANCE * 2` units
- Obstacle avoidance force is weighted at 2x to prioritize safety

### Movement Parameters

```python
MAX_SPEED = 30              # Maximum velocity magnitude
DELTA_T = 0.1              # Time step for physics simulation
INTERACTION_RADIUS = 50    # Distance at which drones sense neighbors
SAFETY_DISTANCE = 10       # Minimum preferred distance between drones
```

### Visualization

- **White dots**: Drones (radius = 5 pixels)
- **Red circles**: Static obstacles
- **Black background**: The 2D environment (800x600)

### Emergent Behavior

Running the simulation produces:
- **Flocking**: The swarm cohesively moves as a unit
- **Obstacle Navigation**: Drones collectively navigate around obstacles
- **Distributed Control**: No bottleneck or single point of failure
- **Scalability**: Works efficiently with 100+ drones using only local interactions

---

## 2. Centralized Simulation (`simulation_centralized.py`)

### Overview
The centralized simulation demonstrates **top-down control** where a single `CentralController` manages all 100 drones globally. The controller computes optimal velocities for each drone based on complete system knowledge.

### Key Features

- **Global Knowledge**: The controller has perfect awareness of all drone positions and target assignments
- **Centralized Task Assignment**: The controller assigns target waypoints to drones
- **Efficient Planning**: Optimal velocity calculations for each drone at every timestep
- **Single Point of Failure**: If the controller fails, all coordination is lost

### Architecture

#### **CentralController Class**
Responsible for:
- Generating target waypoints (20 green dots distributed across the environment)
- Assigning targets to drones (round-robin distribution, 5 drones per target)
- Computing velocities for each drone based on:
  - Direction to assigned target (60% weight)
  - Obstacle avoidance (30% weight)
  - Drone-to-drone collision avoidance (30% weight)
- Reassigning targets when a drone reaches one (within 5 units)

#### **Drone Class (Centralized)**
Simpler than decentralized version:
- Receives velocity commands from the central controller
- Updates position based on assigned velocity
- No local decision-making
```python
def update(self, velocity):
    """Update position with externally assigned velocity"""
```

### Movement Parameters

```python
MAX_SPEED = 2              # Maximum velocity magnitude
DELTA_T = 0.1             # Time step for physics simulation
SAFETY_DISTANCE = 10      # Minimum preferred distance between drones
```

### Visualization

- **White dots**: Drones (radius = 5 pixels)
- **Red circles**: Static obstacles
- **Green dots**: Target waypoints (navigation goals)
- **Black background**: The 2D environment (800x600)

### Control Flow

1. **Initialization**: Controller generates 20 target points
2. **Assignment**: 5 drones per target (100 ÷ 20)
3. **Computation Loop**:
   - For each drone, compute direction to assigned target
   - Add obstacle avoidance forces
   - Add collision avoidance forces
   - Normalize and apply maximum speed limit
4. **Update**: All drones move simultaneously with computed velocities
5. **Reassignment**: When a drone reaches its target (< 5 units), assign next target

---

## Comparison: Decentralized vs. Centralized

| Aspect | Decentralized | Centralized |
|--------|---------------|-------------|
| **Authority** | Distributed (no central control) | Single controller |
| **Scalability** | Excellent (O(n) complexity) | Limited (O(n²) or worse) |
| **Robustness** | High (resilient to failures) | Low (single point of failure) |
| **Efficiency** | Emergent (locally optimal) | Optimal (globally planned) |
| **Communication** | Local only (neighbors within radius) | Global (controller sees all) |
| **Task Complexity** | Simple local rules → complex behavior | Complex central logic |
| **Real-world Applicability** | Bird flocks, insect swarms, biological systems | Air traffic control, warehouse robotics |
| **Latency Issues** | None (local decisions) | May exist (central computation) |
| **Emergence** | Yes, from simple local rules | No, purely algorithmic |

---

## Running the Simulations

### Decentralized:
```bash
python simulation.py
```
Watch the swarm self-organize and navigate obstacles without any centralized instruction!

### Centralized:
```bash
python simulation_centralized.py
```
Watch the centralized controller command drones to visit target points with global coordination.

---

## Project Alignment

These simulations demonstrate:

### Milestone 1: ✅ Completed
- Environmental modeling (2D plane with obstacles)
- Drone representation (position, velocity)
- Local sensing and interaction
- Collision avoidance mechanics

### Milestone 2: ✅ Completed (Decentralized)
- Flocking and formation control (alignment, cohesion, separation)
- Local communication protocols (implicit via neighbor sensing)
- Decentralized coordination

### Milestone 3: 🔄 In Progress
- Task execution and team synergy
- Performance metrics and evaluation
- Mission examples (coverage, target tracking, etc.)

---

## Future Enhancements

### Decentralized:
- Add task allocation (drones divide up regions or targets)
- Implement consensus protocols
- Add dynamic obstacle avoidance
- Performance metrics (coverage %, time-to-completion, collision count)

### Centralized:
- Add communication latency simulation
- Implement health monitoring and drone failure handling
- Add dynamic task prioritization
- Compare computational load with decentralized approach

### Both:
- 3D environment extension
- More complex obstacle configurations
- Performance benchmarking
- Comparison of scalability and efficiency
