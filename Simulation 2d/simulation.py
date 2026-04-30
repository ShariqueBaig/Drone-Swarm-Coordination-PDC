"""
simulation.py -- Pygame 2D Drone Swarm Visualization
PDC Drone Swarm Coordination | 2D Version

KEYS:
    SPACE       Pause / Resume
    R           Reset
    ESC         Quit
    1           Switch to Naive neighbor-finding
    2           Switch to KDTree (quadtree) neighbor-finding
    F           Inject faults (20% drones)
    LClick      Select Mission / Set Waypoint on world
    RClick      Clear Waypoint
"""
import sys
import math
import pygame
import numpy as np
import threading
import time
from collections import deque

import config
from swarm import SwarmManager
from environment import Environment

# ── Visual constants ──────────────────────────────────────────────────────────
DRONE_RADIUS   = 5
HEADING_LEN    = 10
C_BG            = (8, 12, 22)
C_GRID          = (30, 40, 60)
C_BOUNDARY      = (55, 105, 190)
C_DRONE         = (0, 210, 255)
C_DRONE_SEL     = (255, 255, 255)
C_DRONE_DEAD    = (255, 30, 30)
C_DRONE_FAILED  = (255, 140, 30)
C_CARGO         = (255, 165, 30)
C_TASK          = (0, 255, 255)
C_TEXT          = (180, 180, 180)
C_PANEL         = (8, 12, 22, 180)
C_BTN_OFF       = (60, 60, 60, 100)
C_BTN_ON        = (100, 150, 180, 200)
C_COVERAGE      = (0, 210, 255, 40)
C_OBSTACLE      = (215, 35, 55)
C_DYN_OBS       = (180, 50, 200)
C_WAYPOINT      = (0, 255, 120)

class Camera2D:
    """Maps world coordinates (x, y) to screen pixels."""
    def __init__(self, world_w, world_h, screen_w, screen_h):
        self.world_w = world_w
        self.world_h = world_h
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.scale = min(screen_w / world_w, screen_h / world_h) * 0.9
        self.offset = np.array([
            (screen_w - world_w * self.scale) / 2,
            (screen_h - world_h * self.scale) / 2
        ])

    def ws(self, p):
        """World (x, y) → screen (px, py)."""
        return (int(p[0] * self.scale + self.offset[0]),
                int(p[1] * self.scale + self.offset[1]))

    def sw(self, px, py):
        """Screen (px, py) → world (x, y)."""
        wx = (px - self.offset[0]) / self.scale
        wy = (py - self.offset[1]) / self.scale
        return (wx, wy)

    def wl(self, l):
        """World length → screen pixels."""
        return max(1, int(l * self.scale))


def draw_coverage_grid(surface, cam, swarm):
    """Draw 2D coverage heatmap overlay."""
    cell_w = swarm.env.width / swarm.grid_res
    cell_h = swarm.env.height / swarm.grid_res

    overlay = pygame.Surface((surface.get_width(), surface.get_height()), pygame.SRCALPHA)
    visited = np.argwhere(swarm.visited_grid)

    for gx, gy in visited:
        wx = gx * cell_w
        wy = gy * cell_h
        p = cam.ws((wx, wy))
        w = cam.wl(cell_w)
        h = cam.wl(cell_h)
        pygame.draw.rect(overlay, (0, 210, 255, 25), (p[0], p[1], w, h))

    surface.blit(overlay, (0, 0))


def draw_hud_panel(surface, font, swarm, fps, physics_tps, highlighted_mission, btns, method_name):
    # Left Panel
    panel_surf = pygame.Surface((260, 420), pygame.SRCALPHA)
    panel_surf.fill(C_PANEL)
    surface.blit(panel_surf, (10, 10))

    active_count = int(np.sum(~swarm.dead_mask & ~swarm.failed_mask))
    dead_n = int(np.sum(swarm.dead_mask))
    failed_n = int(np.sum(swarm.failed_mask))
    cov = swarm.coverage_pct
    rob = swarm.get_robustness_score(time.time() - start_time)
    track_err = getattr(swarm, "tracking_error_ema", 0.0)
    rec_t = swarm.last_mission_completion_time.get(5, None)
    cov_t = swarm.last_mission_completion_time.get(6, None)
    trn_t = swarm.last_mission_completion_time.get(7, None)
    rec_txt = f"{rec_t:.1f}s" if rec_t is not None else "-"
    cov_txt = f"{cov_t:.1f}s" if cov_t is not None else "-"
    trn_txt = f"{trn_t:.1f}s" if trn_t is not None else "-"

    lines = [
        "DRONE SWARM (2D)",
        "------------------",
        f"Active : {active_count}/{swarm.num_boids}",
        f"Dead   : {dead_n}",
        f"Failed : {failed_n}",
        f"Cover  : {cov:.1f}%",
        f"Collisions: {swarm.collision_count}",
        f"Robust : {rob:.2f}",
        f"TrackErr: {track_err:.1f}",
        f"Done T(s) R/C/T: {rec_txt}/{cov_txt}/{trn_txt}",
        f"Consensus: {getattr(swarm, 'consensus_updates', 0)}",
        f"Method : {method_name.upper()}",
        f"Sim TPS: {int(physics_tps)}",
        f"Render : {int(fps)} FPS",
        "",
        "Controls:",
        "SPACE: Pause   R: Reset",
        "1: Naive  2: KDTree  3: Grid",
        "F: Inject Faults",
        "LClick: Set Waypoint",
        "RClick: Clear Waypoint",
    ]

    for i, line in enumerate(lines):
        txt = font.render(line, True, C_TEXT)
        surface.blit(txt, (20, 25 + i * 20))

    # Right Panel (Missions)
    m_panel = pygame.Surface((220, 310), pygame.SRCALPHA)
    m_panel.fill(C_PANEL)
    surface.blit(m_panel, (surface.get_width() - 230, 10))

    txt = font.render("FLEET COMMAND", True, (150, 150, 150))
    surface.blit(txt, (surface.get_width() - 220, 25))

    mission_labels = ['Idle / Flocking', 'Object Transport', 'Area Coverage', 'Recall Fleet']
    btn_mission_map = [3, 7, 6, 5]

    for i, label in enumerate(mission_labels):
        m_type = btn_mission_map[i]
        color = C_BTN_ON if highlighted_mission == m_type else C_BTN_OFF

        rect = pygame.Rect(surface.get_width() - 215, 60 + i * 60, 190, 45)
        btns[i] = (rect, m_type)

        btn_surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
        btn_surf.fill(color)
        surface.blit(btn_surf, (rect.x, rect.y))
        txt = font.render(label, True, (255, 255, 255))
        surface.blit(txt, (rect.centerx - txt.get_width() // 2, rect.centery - txt.get_height() // 2))

    # Coverage bar at bottom of mission panel
    cov_txt = font.render(f"COVERAGE: {cov:.1f}%", True, (150, 150, 150))
    surface.blit(cov_txt, (surface.get_width() - 220, 310))


def main():
    global start_time
    pygame.init()

    env = Environment()
    swarm = SwarmManager(env)
    start_time = time.time()

    SW, SH = 1280, 720
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("PDC Drone Swarm | 2D Simulation")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 15)

    cam = Camera2D(env.width, env.height, SW, SH)

    highlighted_mission = 3  # Idle by default
    btns = [(None, -1)] * 4
    paused = False
    physics_tps = [0.0]
    show_coverage = True
    waypoint_marker = None

    # ── Physics Thread (Decoupled) ────────────────────────────────────────────
    def physics_worker():
        last_t = time.perf_counter()
        frames = 0
        while True:
            if not paused:
                try:
                    env.reload_if_changed()
                    swarm.update()
                except Exception as e:
                    print(f"[PHYSICS] Error: {e}")
                    time.sleep(0.1)
                    continue
                frames += 1
                time.sleep(0.001)

                curr_t = time.perf_counter()
                if curr_t - last_t >= 1.0:
                    physics_tps[0] = frames / (curr_t - last_t)
                    frames = 0
                    last_t = curr_t
            else:
                time.sleep(0.1)

    t = threading.Thread(target=physics_worker, daemon=True)
    t.start()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    swarm.reset_faults()
                    swarm.mission_type[:] = 3
                    highlighted_mission = 3
                    env.target_waypoint = None
                    waypoint_marker = None
                elif event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_1:
                    swarm.set_method("naive")
                    print("[SIM] Switched to NAIVE")
                elif event.key == pygame.K_2:
                    swarm.set_method("octree")
                    print("[SIM] Switched to KDTREE")
                elif event.key == pygame.K_3:
                    swarm.set_method("grid")
                    print("[SIM] Switched to GRID")
                elif event.key == pygame.K_f:
                    swarm.inject_faults(0.2)
                    print("[SIM] Fault injected: 20%")
                elif event.key == pygame.K_h:
                    show_coverage = not show_coverage
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check mission buttons first
                    clicked_btn = False
                    for btn in btns:
                        if not btn:
                            continue
                        b_rect, m_id = btn
                        if b_rect and b_rect.collidepoint(event.pos):
                            highlighted_mission = m_id
                            swarm.set_fleet_mission(m_id)
                            print(f"[SIM] Fleet Mission: {m_id}")
                            clicked_btn = True
                            break
                    if not clicked_btn:
                        # Set waypoint on world
                        wx, wy = cam.sw(event.pos[0], event.pos[1])
                        if 0 <= wx <= env.width and 0 <= wy <= env.height:
                            env.target_waypoint = np.array([wx, wy])
                            waypoint_marker = (wx, wy)
                            print(f"[SIM] Waypoint set: ({wx:.0f}, {wy:.0f})")
                elif event.button == 3:  # Right click
                    env.target_waypoint = None
                    waypoint_marker = None
                    print("[SIM] Waypoint cleared")

        # ── Rendering ─────────────────────────────────────────────────────────
        screen.fill(C_BG)

        # Grid
        for x in range(0, env.width + 1, 100):
            p1 = cam.ws((x, 0))
            p2 = cam.ws((x, env.height))
            pygame.draw.line(screen, C_GRID, p1, p2, 1)
        for y in range(0, env.height + 1, 100):
            p1 = cam.ws((0, y))
            p2 = cam.ws((env.width, y))
            pygame.draw.line(screen, C_GRID, p1, p2, 1)

        # Boundary
        b1 = cam.ws((0, 0))
        b2 = cam.ws((env.width, env.height))
        pygame.draw.rect(screen, C_BOUNDARY, (b1[0], b1[1], b2[0] - b1[0], b2[1] - b1[1]), 2)

        # Coverage heatmap
        if show_coverage:
            draw_coverage_grid(screen, cam, swarm)

        # Obstacles (static)
        for ob in env.obstacles:
            pos = cam.ws((ob[0], ob[1]))
            r = cam.wl(ob[2])
            pygame.draw.circle(screen, C_OBSTACLE, pos, r, 2)

        # Dynamic Obstacles
        for d in env.dynamic_obstacles:
            pos = cam.ws((d.x, d.y))
            r = cam.wl(d.radius)
            pygame.draw.circle(screen, C_DYN_OBS, pos, r, 2)

        # Waypoint
        if waypoint_marker:
            wp = cam.ws(waypoint_marker)
            pygame.draw.circle(screen, C_WAYPOINT, wp, 8, 2)
            pygame.draw.circle(screen, C_WAYPOINT, wp, 3)

        # ── Snapshot swarm state under lock ────────────────────────────────
        with swarm.state_lock:
            local_pos = swarm.positions.copy()
            local_vel = swarm.velocities.copy()
            local_dead = swarm.dead_mask.copy()
            local_failed = swarm.failed_mask.copy()
            local_tasks = swarm.tasks.copy()
            local_assigned = swarm.assigned_tasks.copy()
            local_mission = swarm.mission_type.copy()
            local_phase = swarm.transport_phase.copy()

        # Task Markers (show only active)
        assigned_count = np.zeros(len(local_tasks))
        for tid in local_assigned:
            if tid != -1:
                assigned_count[tid] += 1

        for tid, tpos in enumerate(local_tasks):
            if tid < len(assigned_count) and assigned_count[tid] > 0:
                p = cam.ws(tpos)
                r = cam.wl(12)
                pts = [(p[0], p[1] - r), (p[0] + r, p[1]), (p[0], p[1] + r), (p[0] - r, p[1])]
                pygame.draw.polygon(screen, C_TASK, pts, 2)
                pygame.draw.circle(screen, (255, 200, 0), p, cam.wl(18), 1)

        # Cargo box
        transporting = (local_mission == 7) & (local_phase == 1)
        if np.any(transporting):
            c_pos = np.mean(local_pos[transporting], axis=0)
            p = cam.ws(c_pos)
            r = cam.wl(15)
            pygame.draw.rect(screen, C_CARGO, (p[0] - r, p[1] - r, r * 2, r * 2), 2)
        else:
            preparing = (local_mission == 7) & (local_phase == 0)
            if np.any(preparing):
                p = cam.ws(local_tasks[8])
                r = cam.wl(15)
                pygame.draw.rect(screen, C_CARGO, (p[0] - r, p[1] - r, r * 2, r * 2), 1)

        # Drones
        for i in range(swarm.num_boids):
            pos = local_pos[i]
            p = cam.ws(pos)

            if local_dead[i]:
                pygame.draw.circle(screen, C_DRONE_DEAD, p, 3)
                continue

            if local_failed[i]:
                pygame.draw.circle(screen, C_DRONE_FAILED, p, 4)
                continue

            is_sel = (highlighted_mission != -1 and local_mission[i] == highlighted_mission)
            color = C_DRONE_SEL if is_sel else C_DRONE
            radius = 5 if is_sel else 4
            pygame.draw.circle(screen, color, p, radius)

            # Heading line
            vel = local_vel[i]
            speed = np.linalg.norm(vel)
            if speed > 0.1:
                nv = vel / speed
                end_p = (int(p[0] + nv[0] * HEADING_LEN), int(p[1] + nv[1] * HEADING_LEN))
                pygame.draw.line(screen, (200, 200, 200), p, end_p, 1)

        # HUD
        draw_hud_panel(screen, font, swarm, clock.get_fps(), physics_tps[0],
                       highlighted_mission, btns, swarm.use_method)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
