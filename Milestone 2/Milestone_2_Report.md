# Milestone 2 Report — Decentralized Drone Swarm Coordination

**PDC Project · Spring 2026**

| | |
|---|---|
| **Course** | Parallel and Distributed Computing |
| **Team** | Sharique Baig (28369) · Suffiyan Asghar Ali (29182) · Muhammad Usman (25177) · Ashhal Aamir (29114) |
| **Milestone** | 2 — Decentralized Coordination Algorithms |
| **Date** | April 20, 2026 |

---

## 1. Introduction

This report documents Milestone 2 of our PDC project: a **100-drone decentralized swarm simulation**. This phase focused on implementing emergent collective behavior (Reynolds flocking, decentralized auctioning) and optimizing the swarm engine for real-time 3D performance.

**Milestone trigger:** Emergent coordinated motion (flocking/formation) visible, dynamic waypoint assignment validated. ✅ Achieved.

---

## 2. System Architecture

### 2.1 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| Physics Engine | NumPy + SciPy | Vectorized kinematics, force accumulation |
| Spatial Indexing | cKDTree / Grid Hash / Naive | Neighbor discovery (O(N log N) / O(N) / O(N²)) |
| Swarm Logic | Python + NumPy | Boids rules, auction protocol, formation control |
| 3D Renderer | Ursina Engine (Panda3D) | Real-time 3D visualization |
| Parallelism | multiprocessing + SharedMemory | Multi-core drone updates |
| Containerization | Docker (Python 3.11-slim) | Reproducible benchmarking |

---

## 3. Team Contributions

### 3.1 Sharique Baig — Swarm Behavioral Logic
- **B2.1–B2.5**: Implemented Reynolds flocking, decentralized auction-based task allocation, consensus-driven conflict resolution, and V-formation control. Integrated local communication radius for pure decentralized sensing.

### 3.2 Suffiyan Asghar Ali — Environment & Physics Engine
- **A2.1–A2.5**: Developed weighted collision steering, predictive avoidance (lookahead), dynamic moving obstacles, and a hot-reload config system. Optimized runtime via obstacle list caching.

### 3.3 Muhammad Usman — Visualization & UI/UX
- **C2.1–C2.5 / C3.1–C3.4**: Developed the 3D Ursina simulation with drone-mesh rendering, mission-state coloring, coverage heatmap, ghost drone interception, and a real-time dashboard for weight tuning.

### 3.4 Ashhal Aamir — Optimization & Parallelism
- **D2.1–D2.5 / D3.1**: Implemented parallel swarm updates (1.76× speedup), shared-memory synchronization, bottleneck profiling, and 3D spatial partitioning (Octree / Grid Hash).

---

## 4. Optimization Results

### 4.1 Physics Engine Performance
Benchmarked using `benchmark_final.py` (300 iterations, 100 drones):

| Optimization | Time | Speedup | Improvement |
|---|---|---|---|
| Original (sequential) | Baseline | 1.00× | — |
| Vectorized (NumPy) | — | **1.19×** | 16% faster |
| Parallel (2 cores) | — | **1.76×** | 44% reduction |

### 4.2 Before vs After (Physics Headless)
| Metric | Before (M1) | After (M2) | Speedup |
|---|---|---|---|
| Physics FPS (headless) | ~12 FPS | **~463 FPS** | **38.6×** |

---

## 5. Spatial Algorithm Comparison (Headless)

Benchmarked using `headless_bench_3d.py` (100 drones, 570 frames):

| Algorithm | Native FPS | Docker FPS | Overhead |
|---|---|---|---|
| **Octree** | 200.6 | 206.3 | -2.8% |
| **Grid Hash** | 120.4 | 128.6 | -6.8% |
| **Naive O(N²)** | 233.4 | 227.9 | +2.4% |

### 5.1 Why Naive O(N²) Wins at N=100
Although the Naive algorithm has the worst theoretical time complexity, it yields the highest FPS for 100 drones due to real-world hardware characteristics:
1. **NumPy Vectorization:** The O(N²) distance matrix is computed using NumPy broadcasting, which executes entirely in highly-optimized, pre-compiled C libraries (BLAS/LAPACK) without any Python-level loops.
2. **Zero Structural Overhead:** Algorithms like Grid Hash and Octree require rebuilding structural representations (grid buckets or cKDTree nodes) every frame. For small values of N, this constant structural overhead is far slower than performing 10,000 raw floating-point calculations.
3. **Cache Locality:** Calculating a dense 100×100 matrix in contiguous memory benefits enormously from CPU hardware pre-fetching, whereas tree and grid lookups involve pointer-chasing and cache misses.

*(Note: If N scaled to 1,000 or 5,000, the Grid and Octree algorithms would easily overtake the Naive method as the $N^2$ math operations would balloon to 25 million+)*

---

## 6. 3D Rendered Performance

### 6.1 Ursina 3D Simulation
The 3D visualization runs natively at **~50 FPS** with 100 drones and all dashboard metrics active.

### 6.2 Docker Container (SW Render)
The Ursina 3D simulation was containerized using Xvfb and Mesa3D. While software rasterization limits the rendered frame rate (~8-12 FPS), the **physics performance remains at ~200 FPS**, confirming portability across environments.

---

## 7. Docker Containerization

A lightweight Docker image (`drone-swarm-bench`) was created for reproducible benchmarks. Results are automatically exported to a `benchmarks/` volume as CSV/JSON.

---

## 8. Evaluation Metrics (Project V2)

| Metric | Target | Achieved | Status |
|---|---|---|---|
| **Mission time** | Minimize | ✅ | Tracked via telemetry |
| **Collision count** | 0 | ✅ | Near-zero with safety margin |
| **Area coverage %** | >95% | ✅ | >99% via voxel grid patrol |
| **Robustness** | >90% | ✅ | Resilient at 25% failure |

---

## 9. Conclusion

Milestone 2 successfully demonstrates **emergent coordinated motion** through purely decentralized algorithms. The system achieves high-fidelity performance (>200 physics FPS) on both native and containerized platforms, laying the foundation for Milestone 3 full-field missions.

---

*Prepared for PDC Course Project, Spring 2026*
*Milestone 2 — Decentralized Coordination Algorithms*
