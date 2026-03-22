import os
import math

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np

class Environment:
    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(os.path.dirname(__file__), "config.yaml")

        # Default world parameters (Milestone 1 requirement)
        self.width = 1000
        self.height = 1000
        self.boundary = "hard-wall"  # or "wrap"
        self.boundary_margin = 20
        self.boundary_repulsion_strength = 1000.0
        self.dt = 0.02
        self.seed = 42
        self.num_drones = 100

        # Default obstacles: (x, y, radius)
        self.obstacles = [(200, 300, 50), (600, 400, 30)]

        self._load_config()

    def _load_config(self):
        if yaml is None:
            return

        if not os.path.exists(self.config_path):
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict):
            return

        world = cfg.get("world", {})
        drone_cfg = cfg.get("drone", {})
        obstacles = cfg.get("obstacles")
        sim_cfg = cfg.get("simulation", {})

        self.width = int(world.get("width", self.width))
        self.height = int(world.get("height", self.height))
        self.boundary = world.get("boundary", self.boundary)
        self.boundary_margin = float(world.get("boundary_margin", self.boundary_margin))
        self.boundary_repulsion_strength = float(world.get("boundary_repulsion_strength", self.boundary_repulsion_strength))

        self.dt = float(sim_cfg.get("dt", self.dt))
        self.seed = int(sim_cfg.get("seed", self.seed))
        self.num_drones = int(sim_cfg.get("num_drones", self.num_drones))

        if obstacles:
            cleaned = []
            for ob in obstacles:
                if not isinstance(ob, (list, tuple)) or len(ob) < 2:
                    continue
                x, y = float(ob[0]), float(ob[1])
                if len(ob) >= 3:
                    r = float(ob[2])
                else:
                    r = 50.0
                cleaned.append((x, y, r))
            if cleaned:
                self.obstacles = cleaned

    def boundary_repulsion(self, position):
        px, py = position
        fx, fy = 0.0, 0.0
        if self.boundary != "hard-wall":
            return np.array([0.0, 0.0], dtype=float)

        margin = float(self.boundary_margin)
        k = float(self.boundary_repulsion_strength)

        if px < margin:
            dist = max(px, 1e-2)
            fx += k / (dist * dist)
        elif px > self.width - margin:
            dist = max(self.width - px, 1e-2)
            fx -= k / (dist * dist)

        if py < margin:
            dist = max(py, 1e-2)
            fy += k / (dist * dist)
        elif py > self.height - margin:
            dist = max(self.height - py, 1e-2)
            fy -= k / (dist * dist)

        return np.array([fx, fy], dtype=float)

    def resolve_boundary(self, position, velocity, delta_t=None):
        px, py = position
        vx, vy = velocity

        if self.boundary == "wrap":
            px = px % self.width
            py = py % self.height
        else:
            if px < 0:
                px = 0
                vx = -vx
            elif px > self.width:
                px = self.width
                vx = -vx
            if py < 0:
                py = 0
                vy = -vy
            elif py > self.height:
                py = self.height
                vy = -vy

        repulse = self.boundary_repulsion((px, py))
        if delta_t is None:
            delta_t = self.dt

        if np.linalg.norm(repulse) > 0:
            vx = vx + repulse[0] * delta_t
            vy = vy + repulse[1] * delta_t

        return (px, py), (vx, vy)

    def clamp_position(self, x, y):
        x = max(0, min(self.width, x))
        y = max(0, min(self.height, y))
        return x, y

    def is_out_of_bounds(self, x, y):
        return x < 0 or y < 0 or x > self.width or y > self.height

    def boundary_bounce(self, position, velocity):
        px, py = position
        vx, vy = velocity
        if px <= 0 or px >= self.width:
            vx = -vx
            px = max(0, min(self.width, px))
        if py <= 0 or py >= self.height:
            vy = -vy
            py = max(0, min(self.height, py))
        return (px, py), (vx, vy)
