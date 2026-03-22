# 2D Swarm Simulation — HUD & Controls Guide
**PDC Project · Milestone 1 · Usman (C1.1–C1.5)**

---

## HUD Panel (top-left)

| Row | What it shows | Why it matters |
|-----|--------------|----------------|
| **FPS** | Frames per second rendered | Target is 60. If this drops below 30 when switching methods, that's your PDC evidence |
| **Drones** | Total drone count (always 100) | Sanity check |
| **Method** | Active neighbor algorithm: NAIVE / GRID / QUADTREE | Shows which of Ashhal's D1.1–D1.3 algorithms is running |
| **Avg neighbors** | Mean neighbor count per drone this tick | Watch this stay roughly constant across methods — proves all three give the same result |
| **Zoom** | Camera zoom as a percentage of default | 100% = full world fits screen. 200% = zoomed in 2× |
| **Neighbor viz** | ON / OFF — whether lines are drawn between neighbors | Toggle with **L** |
| **Time scale** | Physics speed multiplier (0.1× to 2.0×) | Blue = slowed down. Yellow = sped up. White = normal |

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| `SPACE` | **Pause / Resume** simulation |
| `[` | Slow down — decreases time scale by 0.1× (min 0.1×) |
| `]` | Speed up — increases time scale by 0.1× (max 2.0×) |
| `L` | Toggle **neighbor lines** on/off |
| `N` | Switch to **Naive** O(N²) neighbor detection |
| `G` | Switch to **Grid Hash** O(N) neighbor detection |
| `Q` | Switch to **Quadtree** neighbor detection |
| `R` | **Reset camera** to default position and zoom |
| `S` | **Save snapshot** — saves `swarm_snapshot_001.png`, `002.png` etc. in the same folder |
| `ESC` | Quit |

---

## Mouse Controls

| Action | Effect |
|--------|--------|
| **Scroll wheel up** | Zoom in (centred on cursor) |
| **Scroll wheel down** | Zoom out |
| **Middle-click drag** | Pan the camera |
| **Arrow keys** | Pan the camera (held) |
| **Left-click** | Add a new circular obstacle (radius 20) at click position |
| **Right-click** | Remove the obstacle under the cursor |

---

## Banners (centre screen)

| Banner | Meaning |
|--------|---------|
| **PAUSED** (yellow) | Simulation is frozen. Rendering still updates |
| **SLOW 0.5×** (blue) | Simulation running at half speed |

---

## Drone Colors

| Color | Meaning |
|-------|---------|
| **Blue** | Isolated drone — no neighbors within radius R=50 |
| **Green** | Drone in flock — at least one neighbor detected |

Watching the color distribution change as drones cluster is a visual confirmation that neighbor detection is working correctly.

---

## PDC Benchmarking Workflow

This is the recommended sequence to generate evidence for your PDC grade:

1. Run `python src/main.py`
2. Let simulation stabilize for ~5 seconds
3. Press **S** — save baseline screenshot
4. Press **N** — switch to Naive. Note FPS in HUD. Press **S**
5. Press **G** — switch to Grid. Note FPS. Press **S**
6. Press **Q** — switch to Quadtree. Note FPS. Press **S**
7. Run `python src/view_logs.py` — generates FPS graph from `optimized_benchmark.csv`

The FPS difference between NAIVE and GRID/QUADTREE is your O(N²) vs O(N) evidence.

---

## Slow-Motion Use Cases

- Press `[` repeatedly to reach **0.1×** to watch individual drone collision avoidance frame by frame
- Press `[` to **0.5×** to see flocking rules form in slow motion — useful for debugging steering forces
- Press `]` to **2.0×** to quickly see long-term emergent behaviour (crystalline lattice formation)
- Combine with **SPACE** (pause) and **S** (snapshot) for frame-perfect captures
