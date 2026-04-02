"""
environment.py — PDC Project Spring 2026

OPTIMIZATIONS vs original:
────────────────────────────────────────────────────────────────────────────
1. Obstacle cache (obs_centers / obs_radii properties)
   BEFORE: np.array([[ob[0],ob[1]] for ob in self.obstacles]) rebuilt on
           every method call — 6+ times per frame across swarm_optimized.
   AFTER:  Lazy property with dirty-flag. Rebuilt only when obstacles change.
   PDC:    Eliminates 6 × O(M) list traversals from the hot path.

2. resolve_boundary_batch() — vectorized over all N boids
   BEFORE: for i in range(N): pos[i], vel[i] = env.resolve_boundary(...)
   AFTER:  One NumPy call, zero Python-loop overhead.
   PDC:    Data skeleton = MAP. PRAM = EREW.
           SIMD / Array Processor: same operation on N elements at once.
────────────────────────────────────────────────────────────────────────────
"""

import os
try:
    import yaml
except ImportError:
    yaml = None

import numpy as np


class Environment:

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config.yaml")

        self.width  = 1000
        self.height = 1000
        self.boundary = "hard-wall"
        self.boundary_margin = 20
        self.boundary_repulsion_strength = 1000.0
        self.dt         = 0.02
        self.seed       = 42
        self.num_drones = 100

        self.obstacles = [(200, 300, 50), (600, 400, 30)]
        self.dynamic_obstacles = []

        self.hot_reload_enabled = True
        self.reload_check_interval = 30
        self._reload_counter = 0
        self._config_mtime = None

        # Obstacle cache
        self._obs_centers_cache = None
        self._obs_radii_cache   = None
        self._obs_dirty         = True

        self._load_config()
        self._rebuild_obstacle_cache()

    # ── Obstacle cache ────────────────────────────────────────────────────
    def _rebuild_obstacle_cache(self):
        all_obstacles = self.get_all_obstacles()
        if not all_obstacles:
            self._obs_centers_cache = np.empty((0, 2), dtype=np.float64)
            self._obs_radii_cache   = np.empty((0,),   dtype=np.float64)
        else:
            self._obs_centers_cache = np.array(
                [[ob[0], ob[1]] for ob in all_obstacles], dtype=np.float64)
            self._obs_radii_cache   = np.array(
                [ob[2] for ob in all_obstacles], dtype=np.float64)
        self._obs_dirty = False

    @property
    def obs_centers(self):
        if self._obs_dirty:
            self._rebuild_obstacle_cache()
        return self._obs_centers_cache

    @property
    def obs_radii(self):
        if self._obs_dirty:
            self._rebuild_obstacle_cache()
        return self._obs_radii_cache

    def add_obstacle(self, x, y, r=20.0):
        self.obstacles.append((float(x), float(y), float(r)))
        self._obs_dirty = True

    def remove_obstacle(self, index):
        self.obstacles.pop(index)
        self._obs_dirty = True

    def get_all_obstacles(self):
        if not self.dynamic_obstacles:
            return list(self.obstacles)

        dynamic_as_tuples = [
            (ob["x"], ob["y"], ob["r"]) for ob in self.dynamic_obstacles
        ]
        return list(self.obstacles) + dynamic_as_tuples

    def _update_dynamic_obstacles(self, delta_t):
        if not self.dynamic_obstacles:
            return

        moved = False
        for ob in self.dynamic_obstacles:
            x = ob["x"] + ob["vx"] * delta_t
            y = ob["y"] + ob["vy"] * delta_t
            r = ob["r"]

            if x - r < 0:
                x = r
                ob["vx"] *= -1.0
            elif x + r > self.width:
                x = self.width - r
                ob["vx"] *= -1.0

            if y - r < 0:
                y = r
                ob["vy"] *= -1.0
            elif y + r > self.height:
                y = self.height - r
                ob["vy"] *= -1.0

            ob["x"] = x
            ob["y"] = y
            moved = True

        if moved:
            self._obs_dirty = True

    def _update_config_mtime(self):
        if os.path.exists(self.config_path):
            self._config_mtime = os.path.getmtime(self.config_path)

    def _maybe_reload_config(self):
        if not self.hot_reload_enabled or not os.path.exists(self.config_path):
            return

        current_mtime = os.path.getmtime(self.config_path)
        if self._config_mtime is None:
            self._config_mtime = current_mtime
            return

        if current_mtime > self._config_mtime:
            self._load_config()
            self._config_mtime = current_mtime
            self._obs_dirty = True

    def update(self, delta_t=None):
        if delta_t is None:
            delta_t = self.dt

        self._reload_counter += 1
        if self._reload_counter >= self.reload_check_interval:
            self._maybe_reload_config()
            self._reload_counter = 0

        self._update_dynamic_obstacles(delta_t)

    # ── Vectorized batch boundary resolution ─────────────────────────────
    def resolve_boundary_batch(self, positions, velocities, delta_t=None):
        """
        Vectorized boundary resolution for ALL boids in one call.
        Replaces the per-boid Python for-loop in swarm.py / swarm_optimized.py.
        PDC: MAP skeleton, EREW PRAM, SIMD array-processor model.
        """
        if delta_t is None:
            delta_t = self.dt

        if self.boundary == "wrap":
            positions[:, 0] %= self.width
            positions[:, 1] %= self.height
            return positions, velocities

        # X hard-wall
        left  = positions[:, 0] < 0.0
        right = positions[:, 0] > self.width
        positions[left,  0]  = 0.0;          velocities[left,  0] *= -1.0
        positions[right, 0]  = float(self.width); velocities[right, 0] *= -1.0

        # Y hard-wall
        top    = positions[:, 1] < 0.0
        bottom = positions[:, 1] > self.height
        positions[top,    1]  = 0.0;           velocities[top,    1] *= -1.0
        positions[bottom, 1]  = float(self.height); velocities[bottom, 1] *= -1.0

        # Repulsion
        velocities += self._boundary_repulsion_batch(positions) * delta_t
        return positions, velocities

    def _boundary_repulsion_batch(self, positions):
        forces = np.zeros_like(positions)
        if self.boundary != "hard-wall":
            return forces

        margin = float(self.boundary_margin)
        k      = float(self.boundary_repulsion_strength)

        m = positions[:, 0] < margin
        if np.any(m):
            forces[m, 0] += k / np.maximum(positions[m, 0], 1e-2) ** 2

        m = positions[:, 0] > (self.width - margin)
        if np.any(m):
            forces[m, 0] -= k / np.maximum(self.width - positions[m, 0], 1e-2) ** 2

        m = positions[:, 1] < margin
        if np.any(m):
            forces[m, 1] += k / np.maximum(positions[m, 1], 1e-2) ** 2

        m = positions[:, 1] > (self.height - margin)
        if np.any(m):
            forces[m, 1] -= k / np.maximum(self.height - positions[m, 1], 1e-2) ** 2

        return forces

    # ── Original scalar method (kept for legacy / single-boid use) ────────
    def resolve_boundary(self, position, velocity, delta_t=None):
        px, py = position
        vx, vy = velocity

        if self.boundary == "wrap":
            px %= self.width;  py %= self.height
        else:
            if px < 0:             px = 0;            vx = -vx
            elif px > self.width:  px = self.width;   vx = -vx
            if py < 0:             py = 0;            vy = -vy
            elif py > self.height: py = self.height;  vy = -vy

        repulse = self.boundary_repulsion((px, py))
        if delta_t is None:
            delta_t = self.dt
        if np.linalg.norm(repulse) > 0:
            vx += repulse[0] * delta_t
            vy += repulse[1] * delta_t

        return (px, py), (vx, vy)

    def boundary_repulsion(self, position):
        px, py = position
        fx, fy = 0.0, 0.0
        if self.boundary != "hard-wall":
            return np.array([0.0, 0.0], dtype=float)

        margin = float(self.boundary_margin)
        k      = float(self.boundary_repulsion_strength)

        if px < margin:                 fx += k / max(px, 1e-2) ** 2
        elif px > self.width - margin:  fx -= k / max(self.width - px, 1e-2) ** 2
        if py < margin:                 fy += k / max(py, 1e-2) ** 2
        elif py > self.height - margin: fy -= k / max(self.height - py, 1e-2) ** 2

        return np.array([fx, fy], dtype=float)

    # ── Config loading ────────────────────────────────────────────────────
    def _load_config(self):
        if yaml is None or not os.path.exists(self.config_path):
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if not isinstance(cfg, dict):
            return

        world    = cfg.get("world", {})
        sim_cfg  = cfg.get("simulation", {})
        obstacles = cfg.get("obstacles")
        dynamic_obstacles = cfg.get("dynamic_obstacles")

        self.width  = int(world.get("width",  self.width))
        self.height = int(world.get("height", self.height))
        self.boundary = world.get("boundary", self.boundary)
        self.boundary_margin = float(world.get("boundary_margin", self.boundary_margin))
        self.boundary_repulsion_strength = float(
            world.get("boundary_repulsion_strength", self.boundary_repulsion_strength))

        self.dt         = float(sim_cfg.get("dt",         self.dt))
        self.seed       = int(sim_cfg.get("seed",         self.seed))
        self.num_drones = int(sim_cfg.get("num_drones",   self.num_drones))
        self.hot_reload_enabled = bool(
            sim_cfg.get("hot_reload_enabled", self.hot_reload_enabled))
        self.reload_check_interval = int(
            sim_cfg.get("reload_check_interval", self.reload_check_interval))

        if obstacles:
            cleaned = []
            for ob in obstacles:
                if not isinstance(ob, (list, tuple)) or len(ob) < 2:
                    continue
                x, y = float(ob[0]), float(ob[1])
                r    = float(ob[2]) if len(ob) >= 3 else 50.0
                cleaned.append((x, y, r))
            if cleaned:
                self.obstacles = cleaned

        if dynamic_obstacles:
            cleaned_dyn = []
            for ob in dynamic_obstacles:
                if not isinstance(ob, dict):
                    continue
                if not all(k in ob for k in ("x", "y", "r")):
                    continue
                cleaned_dyn.append({
                    "x": float(ob["x"]),
                    "y": float(ob["y"]),
                    "r": float(ob["r"]),
                    "vx": float(ob.get("vx", 0.0)),
                    "vy": float(ob.get("vy", 0.0)),
                })
            self.dynamic_obstacles = cleaned_dyn

        self._update_config_mtime()

    def clamp_position(self, x, y):
        return max(0, min(self.width, x)), max(0, min(self.height, y))

    def is_out_of_bounds(self, x, y):
        return x < 0 or y < 0 or x > self.width or y > self.height

    def boundary_bounce(self, position, velocity):
        px, py = position
        vx, vy = velocity
        if px <= 0 or px >= self.width:
            vx = -vx; px = max(0, min(self.width, px))
        if py <= 0 or py >= self.height:
            vy = -vy; py = max(0, min(self.height, py))
        return (px, py), (vx, vy)
