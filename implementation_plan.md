# Milestone 2 – Performance & UI/UX Overhaul

## Summary

The current simulation runs at **~12 FPS** (per benchmark JSON). The goal is to significantly boost FPS and modernize the UI/UX while maintaining all Milestone 2 behavioral logic (B2.1–B2.5, A2.1–A2.5, D2.1–D2.5).

---

## Root Cause Analysis

### Performance Bottlenecks Found

| Issue | Location | Cost |
|---|---|---|
| `_apply_stuck_escape()` has a Python `for` loop over stuck drone indices — iterates per-drone | `swarm_optimized.py:706` | High |
| `resolve_drone_drone_collisions()` has a Python `for` loop over collision pairs | `swarm_optimized.py:575` | High |
| `resolve_drone_obstacle_collisions()` has a Python `for` loop | `swarm_optimized.py:621` | Medium |
| `find_neighbors_grid()` has nested Python loops to build pairs list | `swarm_optimized.py:122-127` | Medium |
| `_apply_boundary_vectorized()` has 4-iteration Python for-loop over wall configs | `swarm_optimized.py:487` | Low-Medium |
| `swarm.py auction_tasks()` has **O(N²) nested Python loop** (double for-loop over drones) — this is the worst offender in the M2 layer | `swarm.py:226-238` | **Very High** |
| `swarm.py calculate_formation_steer()` has a Python loop over all 100 drones | `swarm.py:277` | High |
| `PerformanceLogger.end_frame()` opens and writes a CSV file every 5 frames (disk I/O in hot path) | `performance_logger.py:54` | Medium |
| `environment.py obstacles` property creates a new list every frame (called 3–4 times per frame) | `environment.py:142` | Low |
| `visualizer.py draw_drones()` iterates with Python for-loop; calls `np.linalg.norm` per-drone | `visualizer.py:198` | Medium |
| `visualizer.py fallback_pairs()` builds full NxN distance matrix every `PAIR_INTERVAL` frames when no `neighbor_mask` | `visualizer.py:153` | Medium |
| `main.py monkey-patch` calls `collision_steer()` which runs full predictive avoidance loop (8 lookahead steps × O(N×M)) every tick | `main.py:59` | High |
| `swarm_optimized.py` imports and instantiates `SpatialGrid` but never actually uses it (cKDTree is used instead) | `swarm_optimized.py:36,61` | Waste |
| `swarm_optimized.py` imports `QuadTree` but never uses it | `swarm_optimized.py:37` | Waste |
| `env.obstacles` property builds a new Python list on every call — called multiple times per frame | `environment.py:136-142` | Low |

### UI/UX Issues

1. Window title still says "M1" — not updated for M2
2. No task zone visualization (C2.3 — color-coded task waypoints with status)
3. No collision counter (C2.5)
4. No formation visualization (C2.2 — formation centroid/target)
5. HUD panel is minimal and plain-looking — no M2 metrics (tasks assigned, tasks complete, formation info)
6. Controls hint bar is hard to read (single long string)
7. No M2-specific hotkeys shown (method switching works but not visible)
8. `show_lines` default = True — drawing 100-drone neighbor lines tanks FPS heavily at startup
9. Clock is capped at 60 but physics is slow — should target 60 FPS with uncapped sim speed
10. No keyboard shortcut for toggling task zone display

---

## Proposed Changes

### 1. `swarm_optimized.py` — Vectorize all Python loops

#### Vectorize `resolve_drone_drone_collisions()`
Replace per-pair Python loop with fully vectorized NumPy scatter operations using `np.add.at`.

#### Vectorize `resolve_drone_obstacle_collisions()`
Replace per-hit Python loop with vectorized position/velocity corrections.

#### Vectorize `_apply_stuck_escape()`
Replace per-stuck-drone Python loop with vectorized kick using `np.where` and broadcasting.

#### Vectorize `_apply_boundary_vectorized()`
Replace the 4-iteration Python for-loop over wall configs with pre-computed stacked arrays.

#### Remove unused imports (`SpatialGrid`, `QuadTree`)
These are imported, instantiated, but never used in the actual computation path (cKDTree handles all three methods). Remove import + instantiation.

#### Optimize `find_neighbors_grid()`
Replace the nested Python `for i, nb in enumerate(nb_lists): for j in nb:` with vectorized logic using `np.concatenate` and `np.repeat`.

---

### 2. `swarm.py` — Vectorize M2 auction loop

#### Vectorize `auction_tasks()`
The current O(N²) Python double-for-loop is the single largest M2 bottleneck. Replace with fully vectorized NumPy conflict resolution using `np.unique` and `np.argmin`.

#### Vectorize `calculate_formation_steer()`
Replace Python for-loop over 100 drones with vectorized index-based offset computation.

---

### 3. `main.py` — Reduce M2 collision steer overhead

The monkey-patched `collision_steer()` runs predictive avoidance (8 lookahead steps × N×M) every single tick. Reduce `pred_lookahead` to 4 steps (sufficient for warning; 8 was excessive) and cache the result every 2 ticks.

---

### 4. `performance_logger.py` — Buffer CSV writes

Currently opens + writes file every 5 frames. Buffer in memory and flush every 100 frames (≈ 1.67s at 60 FPS) to eliminate disk I/O from the hot path.

---

### 5. `environment.py` — Cache obstacle list

Cache `self._obs_cache` and invalidate only when dynamic obstacles move. Eliminates the list construction on every `obstacles` property access (called 3–4× per frame).

---

### 6. `visualizer.py` — Major UI/UX + rendering upgrade

#### FPS & rendering
- Default `show_lines = False` (neighbor lines tank FPS at startup; user can toggle with L)
- Vectorize `draw_drones()` using batch surface blitting instead of per-drone Python loop

#### M2 UI additions (per plan tasks C2.1–C2.5)
- **C2.2**: Draw formation centroid (cross/diamond marker) and V-formation skeleton
- **C2.3**: Draw task waypoints as color-coded circles (Unclaimed=grey, Assigned=yellow, Done=green)
- **C2.5**: Add collision counter to HUD
- **HUD redesign**: Split into two panels — left metrics panel (FPS, drones, method, M2 stats), right task panel (assigned/completed counts, formation status)
- Update window title to "PDC Drone Swarm — Milestone 2"
- Improve font to a cleaner monospace with slightly larger size
- Add subtle grid background pattern for depth

#### New controls
- `T` — toggle task zone display
- `F` — toggle formation visualization

---

## Verification Plan

### Automated
```
cd "Milestone 2/src"
python main.py
```
Monitor HUD FPS counter — target **≥ 30 FPS** (up from ~12 FPS) with all features on.

### Manual
- Confirm all M2 behaviors still visible: drones forming V-shape, moving toward task waypoints, resolving conflicts via auction
- Confirm task zones display with correct color coding
- Confirm HUD shows M2 metrics (tasks assigned/done, collisions)
- Confirm no Python errors in terminal
- Confirm `L` toggles neighbor lines, `T` toggles tasks, `N`/`G`/`Q` switches method

---

## Open Questions

> [!IMPORTANT]
> **Should `swarm.py` (basic) still be kept?** It is no longer the active swarm (main.py uses `swarm_optimized.py`). I will keep it as reference but not change its behavior — only optimize the parts called via `main.py`.

> [!NOTE]
> The `collision_steer()` monkey-patch in `main.py` runs Suffiyan's A2.1/A2.2/A2.5 every tick. This is by design for the integration. I will reduce lookahead from 8→4 steps which halves cost with minimal behavioral change.
