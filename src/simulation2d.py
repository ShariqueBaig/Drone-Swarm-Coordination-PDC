"""
simulation2d.py -- Pygame 2D Version of Milestone 3
Matches behavior and features of simulation3d.py but in 2D.

KEYS:
    SPACE       Pause
    R           Reset
    ESC         Quit
    LClick      Select Mission / Set Waypoint
"""
import sys
import math
import pygame
import numpy as np
import threading
import time
from collections import deque

# Milestone 3 Imports
import config
from swarm_3d import SwarmManager3D
from environment3d import Environment3D

# ── Visual constants ──────────────────────────────────────────────────────────
DRONE_RADIUS   = 5
HEADING_LEN    = 10
C_BG            = (8, 12, 22)
C_GRID          = (20, 60, 100, 100)
C_BOUNDARY      = (55, 105, 190)
C_DRONE         = (0, 210, 255)
C_DRONE_SEL     = (255, 255, 255)
C_DRONE_DEAD    = (255, 30, 30)
C_CARGO         = (255, 165, 30)
C_TASK          = (0, 255, 255)
C_TEXT          = (180, 180, 180)
C_PANEL         = (8, 12, 22, 180)
C_BTN_OFF       = (60, 60, 60, 100)
C_BTN_ON        = (100, 150, 180, 200)

class Camera2D:
    def __init__(self, world_w, world_d, screen_w, screen_h):
        self.world_w = world_w
        self.world_d = world_d
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.scale = min(screen_w/world_w, screen_h/world_d) * 0.9
        self.offset = np.array([(screen_w - world_w*self.scale)/2, (screen_h - world_d*self.scale)/2])

    def ws(self, p):
        # Map 3D (x, z) to 2D screen (x, y)
        return (int(p[0] * self.scale + self.offset[0]),
                int(p[2] * self.scale + self.offset[1]))

    def wl(self, l):
        return max(1, int(l * self.scale))

def draw_hud_panel(surface, font, swarm, fps, physics_tps, highlighted_mission, btns):
    # Left Panel
    panel_surf = pygame.Surface((250, 320), pygame.SRCALPHA)
    panel_surf.fill(C_PANEL)
    surface.blit(panel_surf, (10, 10))
    
    active_count = np.sum(~swarm.dead_mask & ~swarm.failed_mask)
    dead_n = int(np.sum(swarm.dead_mask))
    cov = swarm.coverage_pct
    rob = swarm.get_robustness_score(time.time() - start_time)
    
    lines = [
        "SWARM METRICS (2D)",
        "------------------",
        f"Active : {active_count}/{swarm.num_boids}",
        f"Dead   : {dead_n}",
        f"Cover  : {cov:.1f}%",
        f"Robust : {rob:.2f}",
        f"Sim TPS: {int(physics_tps)}",
        f"Render FPS: {int(fps)}",
        "",
        "Controls:",
        "SPACE: Pause",
        "R: Reset",
        "ESC: Quit"
    ]
    
    for i, line in enumerate(lines):
        txt = font.render(line, True, C_TEXT)
        surface.blit(txt, (20, 25 + i*22))

    # Right Panel (Missions)
    m_panel = pygame.Surface((220, 300), pygame.SRCALPHA)
    m_panel.fill(C_PANEL)
    surface.blit(m_panel, (surface.get_width() - 230, 10))
    
    txt = font.render("FLEET COMMAND", True, (150, 150, 150))
    surface.blit(txt, (surface.get_width() - 220, 25))
    
    mission_labels = ['Idle / Flocking', 'Object Transport', 'Area Coverage', 'Recall Fleet']
    btn_mission_map = [3, 7, 6, 5]
    
    for i, label in enumerate(mission_labels):
        m_type = btn_mission_map[i]
        color = C_BTN_ON if highlighted_mission == m_type else C_BTN_OFF
        
        rect = pygame.Rect(surface.get_width() - 215, 60 + i*60, 190, 45)
        btns[i] = (rect, m_type)
        
        pygame.draw.rect(surface, color, rect, border_radius=5)
        txt = font.render(label, True, (255, 255, 255))
        surface.blit(txt, (rect.centerx - txt.get_width()//2, rect.centery - txt.get_height()//2))

def main():
    global start_time
    pygame.init()
    
    env = Environment3D()
    swarm = SwarmManager3D(env)
    start_time = time.time()
    
    SW, SH = 1280, 720
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("PDC Drone Swarm | Milestone 3 | 2D Replica")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("consolas", 16)
    
    cam = Camera2D(env.width, env.depth, SW, SH)
    
    highlighted_mission = 3 # Idle by default
    btns = [None] * 4
    paused = False
    physics_tps = [0.0]
    
    # ── Physics Thread (Decoupled Replica) ────────────────────────────────────
    def physics_worker():
        last_t = time.perf_counter()
        frames = 0
        while True:
            if not paused:
                swarm.update()
                frames += 1
                # Rate limiting reduced for high-performance throughput
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
                if event.key == pygame.K_SPACE: paused = not paused
                if event.key == pygame.K_r: 
                    swarm.reset_faults()
                    swarm.mission_type[:] = 3
                    highlighted_mission = 3
                if event.key == pygame.K_ESCAPE: running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    # Check mission buttons
                    for b_rect, m_id in btns:
                        if b_rect and b_rect.collidepoint(event.pos):
                            highlighted_mission = m_id
                            alive = ~swarm.dead_mask
                            swarm.mission_type[alive] = m_id
                            swarm.assigned_tasks[alive] = -1
                            if m_id == 7: swarm.transport_phase[alive] = 0
                            print(f"[M3-2D] Fleet Mission Updated: {m_id}")
                            break

        # ── Rendering ─────────────────────────────────────────────────────────
        screen.fill(C_BG)
        
        # Grid
        for x in range(0, env.width + 1, 100):
            p1 = cam.ws((x, 0, 0))
            p2 = cam.ws((x, 0, env.depth))
            pygame.draw.line(screen, (30, 40, 60), p1, p2, 1)
        for z in range(0, env.depth + 1, 100):
            p1 = cam.ws((0, 0, z))
            p2 = cam.ws((env.width, 0, z))
            pygame.draw.line(screen, (30, 40, 60), p1, p2, 1)

        # Boundary
        b1 = cam.ws((0, 0, 0))
        b2 = cam.ws((env.width, 0, env.depth))
        pygame.draw.rect(screen, C_BOUNDARY, (b1[0], b1[1], b2[0]-b1[0], b2[1]-b1[1]), 2)
        
        # Obstacles
        for ob in env.obstacles:
            pos = cam.ws((ob[0], ob[1], ob[2]))
            r = cam.wl(ob[3] // 2)
            pygame.draw.rect(screen, (215, 35, 55), (pos[0]-r, pos[1]-r, r*2, r*2), 1)

        # Dynamic Obstacles
        for d in env.dynamic_obstacles:
            pos = cam.ws((d.x, d.y, d.z))
            r = cam.wl(d.radius // 2)
            pygame.draw.rect(screen, (180, 50, 200), (pos[0]-r, pos[1]-r, r*2, r*2), 1)

        # Task Markers (Only if active)
        with swarm.state_lock:
            local_pos = swarm.positions.copy()
            local_dead = swarm.dead_mask.copy()
            local_tasks = swarm.tasks.copy()
            local_assigned = swarm.assigned_tasks.copy()
            local_mission = swarm.mission_type.copy()
            local_phase = swarm.transport_phase.copy()
        
        assigned_count = np.zeros(len(local_tasks))
        for tid in local_assigned:
            if tid != -1: assigned_count[tid] += 1
            
        for tid, tpos in enumerate(local_tasks):
            if tid < len(assigned_count) and assigned_count[tid] > 0:
                p = cam.ws(tpos)
                r = cam.wl(15)
                # Diamond shape
                pts = [(p[0], p[1]-r), (p[0]+r, p[1]), (p[0], p[1]+r), (p[0]-r, p[1])]
                pygame.draw.polygon(screen, C_TASK, pts, 2)
                pygame.draw.circle(screen, (255, 200, 0), p, cam.wl(20), 1)

        # Cargo
        transporting = (local_mission == 7) & (local_phase == 1)
        if np.any(transporting):
            c_pos = np.mean(local_pos[transporting], axis=0)
            p = cam.ws(c_pos)
            r = cam.wl(20)
            pygame.draw.rect(screen, C_CARGO, (p[0]-r, p[1]-r, r*2, r*2), 2)
        else:
            preparing = (local_mission == 7) & (local_phase == 0)
            if np.any(preparing):
                p = cam.ws(local_tasks[8])
                r = cam.wl(20)
                pygame.draw.rect(screen, C_CARGO, (p[0]-r, p[1]-r, r*2, r*2), 1)

        # Drones
        for i in range(swarm.num_boids):
            p = cam.ws(local_pos[i])
            is_dead = local_dead[i]
            
            if is_dead:
                pygame.draw.circle(screen, C_DRONE_DEAD, p, 3)
            else:
                is_sel = (highlighted_mission != -1 and local_mission[i] == highlighted_mission)
                color = C_DRONE_SEL if is_sel else C_DRONE
                radius = 5 if is_sel else 4
                pygame.draw.circle(screen, color, p, radius)
                
                # Heading
                vel = swarm.velocities[i]
                speed = np.linalg.norm(vel)
                if speed > 0.1:
                    nv = vel / speed
                    end_p = (p[0] + nv[0]*HEADING_LEN, p[1] + nv[2]*HEADING_LEN)
                    pygame.draw.line(screen, (200, 200, 200), p, end_p, 1)

        # HUD
        draw_hud_panel(screen, font, swarm, clock.get_fps(), physics_tps[0], highlighted_mission, btns)
        
        pygame.display.flip()
        # Rendering unlocked (tick 0) for maximum performance
        clock.tick(0)

    pygame.quit()

if __name__ == "__main__":
    main()
