"""
environment.py — Suffiyan's Physics & Environment Engine
PDC Project · Spring 2026

Milestone 1 (A1.1–A1.5): World setup, static obstacles, boundary handling, config loading.
Milestone 2 (A2.1–A2.5):
  A2.1 — Collision Enhancements: weighted steering forces integrated with obstacle avoidance
  A2.2 — Predictive Avoidance:   projects positions forward; steers early if future collision detected
  A2.3 — Dynamic Environment:    moving rectangular obstacles (obs_pos += obs_vel * dt)
  A2.4 — Physics Tuning:         config hot-reload via mtime check (no restart needed)
  A2.5 — Obstacle Sensing:       repulsive force from nearest obstacle edge (not center)
"""

import os
import math
import time

try:
    import yaml
except ImportError:
    yaml = None

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Dynamic obstacle dataclass
# ─────────────────────────────────────────────────────────────────────────────

class DynamicObstacle:
    """
    A2.3 — Moving circular obstacle.

    Attributes
    ----------
    x, y    : current centre position (world units)
    radius  : collision / sensing radius
    vx, vy  : velocity (world units / s)
    """

    def __init__(self, x, y, radius, vx=0.0, vy=0.0):
        self.x      = float(x)
        self.y      = float(y)
        self.radius = float(radius)
        self.vx     = float(vx)
        self.vy     = float(vy)

    # Convenience: unpack as (x, y, r) so existing swarm code stays compatible
    def __iter__(self):
        yield self.x
        yield self.y
        yield self.radius

    def __len__(self):
        return 3

    def __getitem__(self, idx):
        return (self.x, self.y, self.radius)[idx]

    def __repr__(self):
        return (f"DynamicObstacle(pos=({self.x:.1f},{self.y:.1f}), "
                f"r={self.radius}, v=({self.vx:.1f},{self.vy:.1f}))")

    def update(self, dt, world_width, world_height):
        """
        A2.3 — Move obstacle and bounce off world boundaries.

        Formula from spec: obs_pos += obs_vel * dt
        Bounce: reverse velocity component when edge touches the boundary.
        """
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Bounce off left / right walls
        if self.x - self.radius <= 0:
            self.x  = self.radius
            self.vx = abs(self.vx)
        elif self.x + self.radius >= world_width:
            self.x  = world_width - self.radius
            self.vx = -abs(self.vx)

        # Bounce off top / bottom walls
        if self.y - self.radius <= 0:
            self.y  = self.radius
            self.vy = abs(self.vy)
        elif self.y + self.radius >= world_height:
            self.y  = world_height - self.radius
            self.vy = -abs(self.vy)


# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────

class Environment:
    """
    Central environment manager.

    Exposes `self.obstacles` as a unified list of both static and dynamic
    obstacles so that existing swarm / visualizer code needs zero changes.
    Dynamic obstacles are a sub-list (`self.dynamic_obstacles`) that is
    updated each tick via `step(dt)`.
    """

    def __init__(self, config_path=None):
        self.config_path = config_path or os.path.join(
            os.path.dirname(__file__), "config.yaml"
        )

        # ── A1.1 defaults ─────────────────────────────────────────────────────
        self.width                      = 1000
        self.height                     = 1000
        self.boundary                   = "hard-wall"
        self.boundary_margin            = 50
        self.boundary_repulsion_strength = 1000.0
        self.dt                         = 0.02
        self.seed                       = 42
        self.num_drones                 = 100

        # ── A1.2 static obstacles: list of (x, y, r) ─────────────────────────
        self._static_obstacles: list = [(200, 300, 50), (600, 400, 30)]

        # ── A2.3 dynamic obstacles ────────────────────────────────────────────
        self.dynamic_obstacles: list[DynamicObstacle] = []

        # ── A2.4 hot-reload tracking ──────────────────────────────────────────
        self._config_mtime: float = 0.0

        # Initial load
        self._load_config()

    # ──────────────────────────────────────────────────────────────────────────
    # Public property: unified obstacle list consumed by swarm + visualizer
    # ──────────────────────────────────────────────────────────────────────────

    @property
    def obstacles(self):
        """
        Returns static + dynamic obstacles as one flat list.
        Each element unpacks as (x, y, r) — DynamicObstacle supports this.
        """
        return list(self._static_obstacles) + list(self.dynamic_obstacles)

    # ──────────────────────────────────────────────────────────────────────────
    # A2.4 — Config hot-reload
    # ──────────────────────────────────────────────────────────────────────────

    def reload_if_changed(self):
        """
        A2.4 — Check config file mtime and reload if it changed on disk.

        Call this once per simulation tick from main.py.
        Returns True if a reload happened (useful for logging).
        """
        if not os.path.exists(self.config_path):
            return False

        mtime = os.path.getmtime(self.config_path)
        if mtime <= self._config_mtime:
            return False

        print(f"[ENV] Config changed — hot-reloading '{self.config_path}'")
        self._load_config()
        return True

    # ──────────────────────────────────────────────────────────────────────────
    # Config loading (A1.5 extended for M2)
    # ──────────────────────────────────────────────────────────────────────────

    def _load_config(self):
        """
        A1.5 / A2.4 — Load (or reload) config.yaml.
        Parses static obstacles and dynamic obstacle definitions.
        """
        if yaml is None or not os.path.exists(self.config_path):
            return

        with open(self.config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        self._config_mtime = os.path.getmtime(self.config_path)

        if not isinstance(cfg, dict):
            return

        world   = cfg.get("world", {})
        sim_cfg = cfg.get("simulation", {})

        # ── World / simulation parameters ─────────────────────────────────────
        self.width    = int(world.get("width",    self.width))
        self.height   = int(world.get("height",   self.height))
        self.boundary = world.get("boundary",     self.boundary)
        self.boundary_margin = float(
            world.get("boundary_margin", self.boundary_margin)
        )
        self.boundary_repulsion_strength = float(
            world.get("boundary_repulsion_strength", self.boundary_repulsion_strength)
        )
        self.dt         = float(sim_cfg.get("dt",         self.dt))
        self.seed       = int(sim_cfg.get("seed",         self.seed))
        self.num_drones = int(sim_cfg.get("num_drones",   self.num_drones))

        # ── A1.2 Static obstacles ─────────────────────────────────────────────
        raw_static = cfg.get("obstacles")
        if raw_static:
            cleaned = []
            for ob in raw_static:
                if not isinstance(ob, (list, tuple)) or len(ob) < 2:
                    continue
                x, y = float(ob[0]), float(ob[1])
                r    = float(ob[2]) if len(ob) >= 3 else 50.0
                cleaned.append((x, y, r))
            if cleaned:
                self._static_obstacles = cleaned

        # ── A2.3 Dynamic obstacles ────────────────────────────────────────────
        raw_dyn = cfg.get("dynamic_obstacles", [])
        new_dyn = []
        for ob in (raw_dyn or []):
            if not isinstance(ob, (list, tuple, dict)):
                continue
            if isinstance(ob, dict):
                x, y = float(ob.get("x", 0)), float(ob.get("y", 0))
                r    = float(ob.get("radius", 30))
                vx   = float(ob.get("vx", 40))
                vy   = float(ob.get("vy", 30))
            else:
                x, y = float(ob[0]), float(ob[1])
                r    = float(ob[2]) if len(ob) > 2 else 30.0
                vx   = float(ob[3]) if len(ob) > 3 else 40.0
                vy   = float(ob[4]) if len(ob) > 4 else 30.0
            new_dyn.append(DynamicObstacle(x, y, r, vx, vy))

        self.dynamic_obstacles = new_dyn
        print(
            f"[ENV] Loaded: {len(self._static_obstacles)} static, "
            f"{len(self.dynamic_obstacles)} dynamic obstacles."
        )

    # ──────────────────────────────────────────────────────────────────────────
    # A2.3 — Simulation step: update dynamic obstacles
    # ──────────────────────────────────────────────────────────────────────────

    def step(self, dt=None):
        """
        A2.3 — Advance all dynamic obstacles by dt.

        Call once per simulation tick from main.py *before* swarm.update().
        """
        if dt is None:
            dt = self.dt
        for dob in self.dynamic_obstacles:
            dob.update(dt, self.width, self.height)

    # ──────────────────────────────────────────────────────────────────────────
    # A1.4 — Boundary repulsion (unchanged from M1)
    # ──────────────────────────────────────────────────────────────────────────

    def boundary_repulsion(self, position):
        """Outward repulsion force when drone approaches a boundary wall."""
        px, py = position
        fx, fy = 0.0, 0.0

        if self.boundary != "hard-wall":
            return np.array([0.0, 0.0], dtype=float)

        margin = float(self.boundary_margin)
        k      = float(self.boundary_repulsion_strength)

        if px < margin:
            dist = max(px, 1e-2)
            fx  += k / (dist * dist)
        elif px > self.width - margin:
            dist = max(self.width - px, 1e-2)
            fx  -= k / (dist * dist)

        if py < margin:
            dist = max(py, 1e-2)
            fy  += k / (dist * dist)
        elif py > self.height - margin:
            dist = max(self.height - py, 1e-2)
            fy  -= k / (dist * dist)

        return np.array([fx, fy], dtype=float)

    def resolve_boundary(self, position, velocity, delta_t=None):
        """Apply boundary repulsion and hard-clamp on overshoot."""
        px, py = position
        vx, vy = velocity

        if self.boundary == "wrap":
            px = px % self.width
            py = py % self.height
        else:
            if px < 0:
                px, vx = 0, -vx
            elif px > self.width:
                px, vx = self.width, -vx
            if py < 0:
                py, vy = 0, -vy
            elif py > self.height:
                py, vy = self.height, -vy

        repulse = self.boundary_repulsion((px, py))
        if delta_t is None:
            delta_t = self.dt

        if np.linalg.norm(repulse) > 0:
            vx += repulse[0] * delta_t
            vy += repulse[1] * delta_t

        return (px, py), (vx, vy)

    def clamp_position(self, x, y):
        return max(0, min(self.width, x)), max(0, min(self.height, y))

    def is_out_of_bounds(self, x, y):
        return x < 0 or y < 0 or x > self.width or y > self.height

    def boundary_bounce(self, position, velocity):
        px, py = position
        vx, vy = velocity
        if px <= 0 or px >= self.width:
            vx, px = -vx, max(0, min(self.width, px))
        if py <= 0 or py >= self.height:
            vy, py = -vy, max(0, min(self.height, py))
        return (px, py), (vx, vy)

    # ──────────────────────────────────────────────────────────────────────────
    # A2.5 — Obstacle edge sensing: repulsive force from nearest surface point
    # ──────────────────────────────────────────────────────────────────────────

    def obstacle_edge_repulsion(self, positions: np.ndarray,
                                 sense_radius: float = 80.0,
                                 strength: float = 1.6) -> np.ndarray:
        """
        A2.5 — Vectorized repulsive force from the nearest edge of each obstacle.

        Unlike center-based repulsion (which fires too late for large obstacles),
        this measures distance from the *surface* of each obstacle so drones
        receive an earlier, smoother warning signal.

        Force law: F = strength * max_force * (sense_radius - d_surface) / sense_radius
        Direction : radially outward from the nearest surface point.

        Parameters
        ----------
        positions    : (N, 2) array of drone positions
        sense_radius : how far from the obstacle surface the force is active
        strength     : multiplier applied to the resulting force

        Returns
        -------
        forces : (N, 2) repulsive steering forces
        """
        if not self.obstacles:
            return np.zeros((len(positions), 2))

        N = len(positions)
        obs_list = self.obstacles  # includes dynamic obstacles via property
        forces   = np.zeros((N, 2))

        obs_centers = np.array([[ob[0], ob[1]] for ob in obs_list], dtype=float)
        obs_radii   = np.array([ob[2]          for ob in obs_list], dtype=float)

        # Vector from each obstacle centre to each drone: (N, M, 2)
        diff = positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
        # Distance from obstacle centre: (N, M)
        dist_from_center = np.linalg.norm(diff, axis=2)

        # Distance from obstacle *surface*: (N, M)
        dist_from_surface = dist_from_center - obs_radii[np.newaxis, :]

        # Active when within sense_radius of surface and outside the body
        active = (dist_from_surface < sense_radius) & (dist_from_surface > 0)

        if not np.any(active):
            return forces

        # Force magnitude: linear falloff from sense_radius (strong near surface)
        with np.errstate(divide='ignore', invalid='ignore'):
            radial_unit = diff / np.maximum(dist_from_center[:, :, np.newaxis], 1e-9)

        mag = np.where(
            active,
            strength * (sense_radius - dist_from_surface) / sense_radius,
            0.0
        )  # (N, M)

        # Sum contributions from all obstacles
        forces = np.sum(radial_unit * mag[:, :, np.newaxis], axis=1)  # (N, 2)
        return forces

    # ──────────────────────────────────────────────────────────────────────────
    # A2.2 — Predictive avoidance
    # ──────────────────────────────────────────────────────────────────────────

    def predictive_avoidance(self, positions: np.ndarray,
                              velocities: np.ndarray,
                              lookahead_steps: int = 8,
                              k: float = 0.5) -> np.ndarray:
        """
        A2.2 — Project each drone forward and steer away from predicted collisions.

        Formula from spec: pos_pred = pos + vel * k * dt  (extended to multi-step)

        For each lookahead step t ∈ {1, …, lookahead_steps}:
          pos_pred = pos + vel * k * dt * t
          If pos_pred is inside any obstacle, apply an outward avoidance force
          scaled by 1/t (earlier detection → stronger response).

        Parameters
        ----------
        positions       : (N, 2) current drone positions
        velocities      : (N, 2) current drone velocities
        lookahead_steps : number of future positions to check
        k               : fractional dt scale per step (spec default: 0.5)

        Returns
        -------
        avoid_forces : (N, 2) predictive avoidance steering forces
        """
        if not self.obstacles:
            return np.zeros((len(positions), 2))

        N = len(positions)
        obs_centers = np.array([[ob[0], ob[1]] for ob in self.obstacles], dtype=float)
        obs_radii   = np.array([ob[2]          for ob in self.obstacles], dtype=float)

        avoid_forces = np.zeros((N, 2))

        for t in range(1, lookahead_steps + 1):
            # pos_pred = pos + vel * k * dt * t  (spec formula)
            pos_pred = positions + velocities * (k * self.dt * t)   # (N, 2)

            # Distance from each predicted position to each obstacle centre: (N, M)
            diff = pos_pred[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)

            # Collision predicted if projected position enters obstacle body
            collision_r = obs_radii[np.newaxis, :] + 5.0          # +5 = drone body
            predicted_hit = dist < collision_r

            if not np.any(predicted_hit):
                continue

            # Avoidance direction: radially away from obstacle centre
            with np.errstate(divide='ignore', invalid='ignore'):
                radial_unit = diff / np.maximum(dist[:, :, np.newaxis], 1e-9)

            # Scale by 1/t: imminent collision gets stronger response
            step_weight = 1.0 / t
            mag = np.where(predicted_hit, step_weight, 0.0)        # (N, M)

            avoid_forces += np.sum(radial_unit * mag[:, :, np.newaxis], axis=1)

        return avoid_forces

    # ──────────────────────────────────────────────────────────────────────────
    # A2.1 — Collision enhancement: weighted steering force helper
    # ──────────────────────────────────────────────────────────────────────────

    def collision_steer(self, positions: np.ndarray,
                         velocities: np.ndarray,
                         max_speed: float,
                         max_force: float,
                         edge_sense_radius: float = 80.0,
                         edge_strength:     float = 1.6,
                         pred_lookahead:    int   = 8,
                         pred_k:            float = 0.5,
                         pred_weight:       float = 1.2) -> np.ndarray:
        """
        A2.1 — Combined collision avoidance steering.

        Merges A2.5 (edge repulsion) and A2.2 (predictive avoidance) into one
        weighted steering vector compatible with the Boids acceleration pipeline.

        Force composition:
          total = edge_repulsion  (A2.5)
                + predictive * pred_weight  (A2.2)

        The result is clamped to max_force so it integrates cleanly alongside
        Boids separation / alignment / cohesion forces.

        Parameters
        ----------
        positions / velocities : (N, 2) drone state arrays
        max_speed / max_force  : from config — used to normalise steering
        edge_sense_radius      : A2.5 activation distance from surface
        edge_strength          : A2.5 force multiplier
        pred_lookahead         : A2.2 number of future steps to probe
        pred_k                 : A2.2 fractional dt per step
        pred_weight            : relative weight of predictive vs edge force

        Returns
        -------
        steer : (N, 2) clamped collision steering force
        """
        edge_f = self.obstacle_edge_repulsion(positions, edge_sense_radius, edge_strength)
        pred_f = self.predictive_avoidance(positions, velocities, pred_lookahead, pred_k)

        combined = edge_f + pred_f * pred_weight

        # Normalise to desired_velocity then cap at max_force (standard Boids steer)
        mags  = np.linalg.norm(combined, axis=1, keepdims=True)
        valid = np.atleast_1d((mags > 1e-9).squeeze())
        steer = np.zeros_like(combined)
        if np.any(valid):
            steer[valid] = (combined[valid] / mags[valid]) * max_speed
            steer[valid] -= velocities[valid]
            fmags = np.linalg.norm(steer[valid], axis=1, keepdims=True)
            over  = np.atleast_1d(fmags.squeeze()) > max_force
            if np.any(over):
                idx = np.where(valid)[0][over]
                steer[idx] = (steer[idx] / fmags[over]) * max_force

        return steer