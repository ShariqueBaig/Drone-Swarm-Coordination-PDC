"""
swarm.py — Base SwarmManager, PDC Project Spring 2026

OPTIMIZATION vs original:
────────────────────────────────────────────────────────────────────────────
Boundary loop (original line 124):
  BEFORE: for i in range(self.num_boids):
              new_pos, new_vel = env.resolve_boundary(pos[i], vel[i], dt)
  AFTER:  self.positions, self.velocities = env.resolve_boundary_batch(...)
  PDC:    MAP skeleton. EREW PRAM. SIMD: N boids processed in one call.

Obstacle arrays pre-extracted outside loop (minor, avoids double list comp).
────────────────────────────────────────────────────────────────────────────
"""

import numpy as np
try:
    import pygame
except ImportError:
    pass
import config


class SwarmManager:

    def __init__(self, env):
        self.env = env
        np.random.seed(env.seed)
        self.num_boids     = env.num_drones
        self.ids           = np.arange(self.num_boids)
        self.positions     = np.random.rand(self.num_boids, 2) * [env.width, env.height]
        self.velocities    = (np.random.rand(self.num_boids, 2) - 0.5) * config.max_speed
        self.accelerations = np.zeros((self.num_boids, 2))

    def update(self, dt=None):
        if dt is None:
            dt = config.dt

        self.accelerations = np.zeros((self.num_boids, 2))

        # Distance matrix (N, N, 2)
        diff_matrix = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=2)
        np.fill_diagonal(dist_matrix, np.inf)

        mask            = dist_matrix < config.perception_radius
        separation_mask = dist_matrix < config.safety_distance

        # Separation
        with np.errstate(divide='ignore', invalid='ignore'):
            separation_vectors = diff_matrix / dist_matrix[:, :, np.newaxis]
        separation_vectors[~separation_mask] = 0
        separation = np.sum(separation_vectors, axis=1)

        # Alignment
        neighbor_velocities = self.velocities[np.newaxis, :, :]
        alignment_sum       = np.sum(neighbor_velocities * mask[:, :, np.newaxis], axis=1)
        neighbor_counts     = np.sum(mask, axis=1)[:, np.newaxis]

        # Cohesion
        neighbor_positions = self.positions[np.newaxis, :, :]
        cohesion_sum       = np.sum(neighbor_positions * mask[:, :, np.newaxis], axis=1)

        sep_steer = self.steer(separation, 0)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_vel = alignment_sum / neighbor_counts
        avg_vel[np.isnan(avg_vel)] = 0
        align_steer = self.steer(avg_vel, 0, subtract_velocity=True)

        with np.errstate(divide='ignore', invalid='ignore'):
            avg_pos = cohesion_sum / neighbor_counts
        avg_pos[np.isnan(avg_pos)] = 0

        vec_to_com = avg_pos - self.positions
        vec_to_com[neighbor_counts.flatten() == 0] = 0
        cohesion_steer = self.steer(vec_to_com, 0, subtract_velocity=True)

        # Obstacle avoidance
        obstacle_steer = np.zeros((self.num_boids, 2))
        if self.env.obstacles:
            obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
            obs_radii   = np.array([ob[2] for ob in self.env.obstacles])

            obs_diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
            obs_dist = np.linalg.norm(obs_diff, axis=2)
            obs_mask = obs_dist < (config.perception_radius + obs_radii[np.newaxis, :])

            with np.errstate(divide='ignore', invalid='ignore'):
                obs_avoid_vec = (obs_diff / obs_dist[:, :, np.newaxis]) / (
                    obs_dist[:, :, np.newaxis] / obs_radii[np.newaxis, :, np.newaxis])

            obs_avoid_vec[~obs_mask] = 0
            obstacle_steer = np.sum(obs_avoid_vec, axis=1)
            obstacle_steer = self.steer(obstacle_steer, 0)

        self.accelerations += sep_steer   * config.separation_weight
        self.accelerations += align_steer * config.alignment_weight
        self.accelerations += cohesion_steer * config.cohesion_weight
        self.accelerations += obstacle_steer * config.obstacle_weight

        self.velocities += self.accelerations

        speeds     = np.linalg.norm(self.velocities, axis=1)
        limit_mask = speeds > config.max_speed
        self.velocities[limit_mask] = (
            self.velocities[limit_mask] / speeds[limit_mask, np.newaxis]) * config.max_speed

        self.positions += self.velocities * dt

        # ── OPTIMIZATION: vectorized boundary — replaces Python for-loop ──
        # BEFORE: for i in range(self.num_boids):
        #             new_pos, new_vel = env.resolve_boundary(pos[i], vel[i])
        # AFTER:  one NumPy call, N boids processed simultaneously.
        self.positions, self.velocities = self.env.resolve_boundary_batch(
            self.positions, self.velocities, dt)

        self.resolve_collisions()

    def resolve_collisions(self):
        if not self.env.obstacles:
            return

        obs_centers = np.array([[ob[0], ob[1]] for ob in self.env.obstacles])
        obs_radii   = np.array([ob[2] for ob in self.env.obstacles])

        diff = self.positions[:, np.newaxis, :] - obs_centers[np.newaxis, :, :]
        dist = np.linalg.norm(diff, axis=2)

        collision_radius = obs_radii[np.newaxis, :] + 5
        collisions       = dist < collision_radius

        if np.any(collisions):
            boid_indices, obs_indices = np.where(collisions)
            for boid_idx, obs_idx in zip(boid_indices, obs_indices):
                vec = self.positions[boid_idx] - obs_centers[obs_idx]
                d   = np.linalg.norm(vec)
                if d == 0:
                    vec = np.random.randn(2);  d = np.linalg.norm(vec)
                normal  = vec / d
                overlap = collision_radius[0, obs_idx] - d
                self.positions[boid_idx] += normal * overlap
                v   = self.velocities[boid_idx]
                dot = np.dot(v, normal)
                if dot < 0:
                    self.velocities[boid_idx] = v - 2 * dot * normal

    def steer(self, vectors, target=None, subtract_velocity=False):
        magnitudes = np.linalg.norm(vectors, axis=1)
        valid      = magnitudes > 0

        result        = np.zeros_like(vectors)
        result[valid] = (vectors[valid] / magnitudes[valid, np.newaxis]) * config.max_speed

        if subtract_velocity:
            result[valid] -= self.velocities[valid]

        force_mags  = np.linalg.norm(result, axis=1)
        limit_force = force_mags > config.max_force
        result[limit_force] = (
            result[limit_force] / force_mags[limit_force, np.newaxis]) * config.max_force

        return result
