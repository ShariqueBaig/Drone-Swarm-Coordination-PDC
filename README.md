# Drone Swarm Coordination: Parallel & Distributed Simulation

A high-performance, decentralized coordination system enabling large swarms of autonomous drones to perform complex missions through emergent behavior and localized interactions.

## Team Members
- **Sharique Baig**
- **Suffiyan**
- **Ashhal**
- **Usman**

---

## Project Overview
This project simulates 100+ autonomous drones in a shared 3D environment, focusing on maximizing scalability and efficiency through Parallel and Distributed Computing (PDC) techniques. The system demonstrates how simple local rules can lead to complex global coordination without the need for a centralized bottleneck.

### Core Implementation
The project evolved through three major milestones:
1.  **Foundational 2D Simulation**: Establishing basic kinematics, reactive collision avoidance, and static environment modeling.
2.  **Optimized 3D Visualization**: Transitioning to a full 3D environment with high-fidelity rendering (Ursina Engine) and vectorized performance optimizations.
3.  **HPC & Parallel Performance**: Integrating spatial indexing (Octree), asynchronous GPU compute pipelines, and multi-threaded execution to handle extreme agent counts.

### Key Features
- **Decentralized Flocking**: Reynolds-style behaviors (Separation, Alignment, Cohesion) implemented using optimized NumPy vectorization to minimize Python loop overhead.
- **Distributed Task Allocation**: An asynchronous auction protocol where drones bid on tasks based on local proximity and consensus resolution, ensuring no two drones fight over the same objective.
- **Spatial Partitioning (KD-Tree/Octree)**: Advanced spatial indexing that reduces neighbor-finding complexity from $O(N^2)$ to $O(N \log N)$, enabling real-time simulation of thousands of drones.
- **Dynamic Mission Logic**:
  - **Area Coverage**: Drones track their own paths to ensure 99%+ coverage of the search volume, visualized through a dynamic floor heatmap.
  - **Coordinated Transport**: Specialized formation logic for multiple drones to synchronize and carry heavy objects to target drop-off points.
  - **Fault Injection**: Real-time simulation of drone attrition (failures) and how the swarm dynamically adapts to fill gaps in the formation.

---

## Performance Analysis

### Scaling Success
By implementing the `cKDTree` algorithm, the simulation achieves a massive speedup as the drone count scales, proving the efficiency of spatial indexing over naive pairwise distance calculations.

| Drone Count | Naive Method | Octree Method | Speedup |
|-------------|--------------|---------------|---------|
| 500 drones  | 299.43 ms    | 11.59 ms      | **25.8x** |
| 1,000 drones| 888.71 ms    | 15.14 ms      | **58.7x** |
| 1,500 drones| 2020.78 ms   | 22.16 ms      | **91.2x** |

## High-Performance Computing (HPC) & Parallel Techniques

To achieve real-time performance for 100+ agents in a 3D environment, the simulation implements several key HPC strategies:

### 1. Advanced Spatial Partitioning (Octree/KD-Tree)
The primary bottleneck in swarm simulations is the $O(N^2)$ neighbor-search problem.
- **Implementation**: We utilize `scipy.spatial.cKDTree` to create a dynamic spatial index every frame.
- **Benefit**: This reduces the complexity to $O(N \log N)$, allowing the simulation to scale to 2,000+ drones while maintaining high frame rates.

### 2. Data Parallelism & Vectorization
- **SIMD Operations**: Using NumPy's vectorized operations, we offload element-wise calculations (like distance and force accumulation) to highly optimized C and Fortran backends.
- **Memory Alignment**: All drone states are stored in contiguous `np.float64` arrays to ensure optimal cache line utilization and minimize memory fetch overhead.

### 3. Task Parallelism (Multi-threading)
- **Fork-Join Model**: The simulation update loop is decomposed into independent tasks (neighbor search, force calculation, mission logic).
- **Concurrency**: Using `ThreadPoolExecutor`, these tasks are distributed across available CPU cores, maximizing throughput during the most computationally expensive phases of the simulation.

### 4. GPU-Accelerated Pipeline (Experimental)
- **Async CUDA Streams**: Support for `CuPy` allows for asynchronous data transfer between CPU and GPU, enabling massive pairwise calculations to be offloaded to the GPU without blocking the main rendering thread.

### 5. Amdahl's & Gustafson's Law Validation
The project includes a built-in `ParallelMetrics` suite that calculates the **Parallel Fraction ($P$)** of the simulation. By measuring serial overhead (rendering, I/O) vs parallelizable compute, we ensure the system adheres to theoretical speedup models.

---

## Execution Guide

### 1. Installation
Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

### 2. Running Simulations
- **Latest 3D Simulation (Root Source)**:
  ```bash
  python src/simulation3d.py
  ```
- **Latest 2D Simulation (Root Simulation 2d)**:
  ```bash
  python "Simulation 2d/simulation.py"
  ```

### 3. Benchmarking & Analysis
- **Neighbor Search Benchmark**:
  ```bash
  python src/benchmark_neighbor_algos.py
  ```
- **Visualize Performance Graphs**:
  ```bash
  python src/plot_neighbor_benchmark.py
  ```

---

## Repository Structure
- `src/`: Core 3D simulation engine, GPU pipelines, and spatial optimization logic.
- `Simulation 2d/`: Specialized 2D coordination, gossip protocol, and local consensus simulation.
- `Milestone 1-3/`: Documentation and snapshots of each development phase.
- `docs/`: Technical research and project guidelines.
- `parallel_analysis.csv`, `swarm_evaluation.csv`, `m2_performance.csv`: Key performance datasets.

---

## Keyboard Controls (3D Sim)
| Key | Action |
|-----|--------|
| `1` / `2` | Switch Algorithm: Naive vs Octree |
| `B` | Toggle Performance HUD (FPS/Speedup) |
| `H` | Toggle Coverage Heatmap |
| `T` | Toggle Drone Trails |
| `O` | Obstacle Placement Mode |
| `C` | Cinematic Camera Mode |
| `R` | Reset Simulation |
