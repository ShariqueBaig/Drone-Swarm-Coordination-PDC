"""
visualizer.py — Usman's module · PDC Project · Milestone 2
Tasks: C1.1-C1.5 (M1) + C2.1-C2.5 (M2)

M2 additions:
  C2.2 — Formation centroid + V-skeleton
  C2.3 — Task zone colour-coding (Unclaimed/Assigned/Done)
  C2.5 — Collision counter in HUD

CONTROLS:
    SPACE       pause / resume
    [ / ]       slow / fast
    L           toggle neighbor lines  (default OFF for FPS)
    T           toggle task zones
    F           toggle formation viz
    N/G/Q       switch neighbor method
    R           reset camera
    S           snapshot
    Scroll      zoom  |  Arrows/Mid-drag  pan
    LClick      add obstacle  |  RClick remove obstacle
    ESC         quit
"""
import sys
import math
import pygame
import numpy as np
import config

# ── Visual constants ──────────────────────────────────────────────────────────
DRONE_RADIUS   = 4
HEADING_LEN    = 12
PAN_SPEED      = 10
NEW_OBS_RADIUS = 20
PAIR_INTERVAL  = 6          # recompute neighbor pairs every N frames (was 4)

TIME_SCALE_MIN  = 0.1
TIME_SCALE_MAX  = 2.0
TIME_SCALE_STEP = 0.1

# Palette
C_BG            = (12,  16,  26)
C_GRID          = (22,  28,  42)
C_BOUNDARY      = (45,  55,  80)
C_DRONE_SOLO    = (0,  180, 255)
C_DRONE_FLOCK   = (40, 220, 130)
C_DRONE_DEAD    = (220, 40,  40)
C_DRONE_DEAD_BD = (120, 10,  10)
C_DRONE_BORDER  = (10,  70,  55)
C_HEADING       = (255, 255, 255)
C_NEIGHBOR_LINE = (35,  170, 120)
C_OBS_FILL      = (50,  22,  22)
C_OBS_BORDER    = (210, 65,  40)
C_OBS_STRIPE    = (90,  32,  22)
C_OBS_HIGHLIGHT = (240, 110, 70)
C_TEXT          = (210, 215, 230)
C_TEXT_DIM      = (90,  95, 115)
C_TEXT_WARN     = (255, 200, 50)
C_TEXT_OK       = (60,  220, 130)
C_PAUSED        = (255, 200, 50)
C_SLOWMO        = (100, 200, 255)
C_PANEL         = (0,   0,   0,  175)

# Task zone colours
C_TASK_UNCLAIMED = (100, 100, 120)
C_TASK_ASSIGNED  = (255, 200,  50)
C_TASK_DONE      = (60,  220, 130)
C_FORMATION_CTR  = (180, 100, 255)
C_FORMATION_LINE = (120,  70, 200)


# ── Camera ────────────────────────────────────────────────────────────────────
class Camera:
    def __init__(self, world_w, world_h, screen_w, screen_h):
        self.world_w  = world_w;  self.world_h  = world_h
        self.screen_w = screen_w; self.screen_h = screen_h
        self.reset()

    def reset(self):
        self.scale  = min(self.screen_w / self.world_w,
                          self.screen_h / self.world_h)
        self.offset = np.array([0.0, 0.0])

    def ws(self, p):
        return (int((p[0] + self.offset[0]) * self.scale),
                int((p[1] + self.offset[1]) * self.scale))

    def wl(self, l):
        return max(1, int(l * self.scale))

    def zoom(self, factor, pivot):
        pw = np.array(pivot, dtype=float) / self.scale - self.offset
        self.scale  = max(0.15, min(8.0, self.scale * factor))
        self.offset = np.array(pivot, dtype=float) / self.scale - pw

    def pan(self, delta):
        self.offset += np.array(delta, dtype=float) / self.scale

    def s2w(self, sp):
        return (sp[0] / self.scale - self.offset[0],
                sp[1] / self.scale - self.offset[1])

    @property
    def zoom_pct(self):
        base = min(self.screen_w / self.world_w, self.screen_h / self.world_h)
        return int(self.scale * 100 / base)


# ── Obstacle surface cache ────────────────────────────────────────────────────
_obs_cache: dict = {}

def _make_obs_surf(sr):
    size = sr * 2
    s = pygame.Surface((size, size), pygame.SRCALPHA)
    gap = max(6, sr // 4)
    for x in range(-size, size, gap):
        pygame.draw.line(s, (*C_OBS_STRIPE, 180), (x, 0), (x + size, size), 1)
    m = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.circle(m, (255, 255, 255, 255), (sr, sr), sr)
    s.blit(m, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return s

def draw_obstacles(surface, obstacles, camera):
    global _obs_cache
    sk = round(camera.scale, 2)
    new: dict = {}
    for ox, oy, r in obstacles:
        sx, sy = camera.ws((ox, oy))
        sr = camera.wl(r)
        if sr < 2: continue
        key = (round(ox), round(oy), round(r), sk)
        surf = _obs_cache.get(key) or _make_obs_surf(sr)
        new[key] = surf
        pygame.draw.circle(surface, C_OBS_FILL, (sx, sy), sr)
        surface.blit(surf, (sx - sr, sy - sr))
        pygame.draw.circle(surface, C_OBS_BORDER, (sx, sy), sr, 2)
        pygame.draw.arc(surface, C_OBS_HIGHLIGHT,
                        (sx-sr, sy-sr, sr*2, sr*2), math.pi*.75, math.pi*1.5, 2)
    _obs_cache = new


# ── Background grid ───────────────────────────────────────────────────────────
def draw_grid(surface, env, camera, grid_step=100):
    for x in range(0, env.width + 1, grid_step):
        sx1, sy1 = camera.ws((x, 0))
        sx2, sy2 = camera.ws((x, env.height))
        pygame.draw.line(surface, C_GRID, (sx1, sy1), (sx2, sy2), 1)
    for y in range(0, env.height + 1, grid_step):
        sx1, sy1 = camera.ws((0, y))
        sx2, sy2 = camera.ws((env.width, y))
        pygame.draw.line(surface, C_GRID, (sx1, sy1), (sx2, sy2), 1)


def draw_boundary(surface, env, camera):
    x1, y1 = camera.ws((0, 0))
    x2, y2 = camera.ws((env.width, env.height))
    pygame.draw.rect(surface, C_BOUNDARY, (x1, y1, x2-x1, y2-y1), 2)


# ── C2.3 Task zones ───────────────────────────────────────────────────────────
def draw_task_zones(surface, swarm_mgr, camera):
    tasks = getattr(swarm_mgr, 'tasks', None)
    if tasks is None or len(tasks) == 0:
        return
    assigned = getattr(swarm_mgr, 'assigned_tasks', None)
    task_radius = getattr(config, 'task_radius', 20)

    for t_idx, (tx, ty) in enumerate(tasks):
        sx, sy = camera.ws((tx, ty))
        sr = max(2, camera.wl(task_radius))

        # Determine status
        if assigned is not None and np.any(assigned == t_idx):
            color = C_TASK_ASSIGNED
            label = "A"
        else:
            color = C_TASK_UNCLAIMED
            label = "?"

        # Outer glow ring
        glow_surf = pygame.Surface((sr*4, sr*4), pygame.SRCALPHA)
        pygame.draw.circle(glow_surf, (*color, 30), (sr*2, sr*2), sr*2)
        surface.blit(glow_surf, (sx - sr*2, sy - sr*2))

        pygame.draw.circle(surface, (*color, 80), (sx, sy), sr)
        pygame.draw.circle(surface, color, (sx, sy), sr, 2)

        # Cross-hair
        ch = max(4, sr // 2)
        pygame.draw.line(surface, color, (sx-ch, sy), (sx+ch, sy), 1)
        pygame.draw.line(surface, color, (sx, sy-ch), (sx, sy+ch), 1)


# ── C2.2 Formation visualizer ─────────────────────────────────────────────────
def draw_formation(surface, swarm_mgr, camera):
    positions  = swarm_mgr.positions
    velocities = swarm_mgr.velocities
    dead_mask  = getattr(swarm_mgr, 'dead_mask', None)

    alive = ~dead_mask if dead_mask is not None else np.ones(len(positions), dtype=bool)
    if not np.any(alive):
        return

    centroid   = np.mean(positions[alive], axis=0)
    vel_cent   = np.mean(velocities[alive], axis=0)
    speed      = np.linalg.norm(vel_cent)

    cx, cy = camera.ws(centroid)

    # Draw centroid marker
    r = camera.wl(8)
    pygame.draw.circle(surface, C_FORMATION_CTR, (cx, cy), r, 2)
    pygame.draw.line(surface, C_FORMATION_CTR, (cx-r, cy), (cx+r, cy), 1)
    pygame.draw.line(surface, C_FORMATION_CTR, (cx, cy-r), (cx, cy+r), 1)

    # Draw heading arrow from centroid
    if speed > 1.0:
        dir_vec = vel_cent / speed
        arr_len = camera.wl(40)
        ex = int(cx + dir_vec[0] * arr_len)
        ey = int(cy + dir_vec[1] * arr_len)
        pygame.draw.line(surface, C_FORMATION_CTR, (cx, cy), (ex, ey), 2)


# ── Neighbor lines ────────────────────────────────────────────────────────────
def pairs_from_mask(mask):
    p = np.argwhere(mask)
    return p[p[:, 0] < p[:, 1]]

def fallback_pairs(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    p = np.argwhere(dist < config.perception_radius)
    return p[p[:, 0] < p[:, 1]]

def draw_neighbor_lines(surface, positions, pairs, camera, show):
    if not show or len(pairs) == 0: return
    sw, sh = surface.get_size()
    for i, j in pairs:
        sx1, sy1 = camera.ws(positions[i])
        sx2, sy2 = camera.ws(positions[j])
        if (max(sx1,sx2)<-50 or min(sx1,sx2)>sw+50 or
                max(sy1,sy2)<-50 or min(sy1,sy2)>sh+50): continue
        pygame.draw.line(surface, C_NEIGHBOR_LINE, (sx1,sy1), (sx2,sy2), 1)


# ── Drone rendering ───────────────────────────────────────────────────────────
def draw_drones(surface, positions, velocities, neighbor_counts, dead_mask, camera):
    r        = camera.wl(DRONE_RADIUS)
    head_len = camera.wl(HEADING_LEN)
    sw, sh   = surface.get_size()
    has_nc   = neighbor_counts is not None
    has_dead = dead_mask is not None

    for i in range(len(positions)):
        sx, sy = camera.ws(positions[i])
        if not (-r*2 <= sx <= sw+r*2 and -r*2 <= sy <= sh+r*2):
            continue
        is_dead = has_dead and dead_mask[i]
        if is_dead:
            pygame.draw.circle(surface, C_DRONE_DEAD, (sx, sy), max(r, 3))
            pygame.draw.circle(surface, C_DRONE_DEAD_BD, (sx, sy), max(r, 3), 1)
            cx = max(r-1, 2)
            pygame.draw.line(surface, C_DRONE_DEAD_BD, (sx-cx, sy-cx), (sx+cx, sy+cx), 2)
            pygame.draw.line(surface, C_DRONE_DEAD_BD, (sx+cx, sy-cx), (sx-cx, sy+cx), 2)
        else:
            color = C_DRONE_FLOCK if (has_nc and neighbor_counts[i] > 0) else C_DRONE_SOLO
            pygame.draw.circle(surface, color, (sx, sy), max(r, 2))
            if r > 2:
                pygame.draw.circle(surface, C_DRONE_BORDER, (sx, sy), r, 1)
            speed = np.linalg.norm(velocities[i])
            if speed > 0.01:
                nv = velocities[i] / speed
                pygame.draw.line(surface, C_HEADING, (sx, sy),
                                 (int(sx + nv[0]*head_len), int(sy + nv[1]*head_len)), 1)


# ── HUD ───────────────────────────────────────────────────────────────────────
def _panel(surface, x, y, w, h):
    s = pygame.Surface((w, h), pygame.SRCALPHA)
    s.fill(C_PANEL)
    surface.blit(s, (x, y))

def draw_hud(surface, font_sm, font_lg, clock, swarm_mgr,
             paused, show_lines, show_tasks, show_formation,
             time_scale, camera, sw, sh):

    method   = getattr(swarm_mgr, 'use_method',      'N/A')
    avg_nb   = getattr(swarm_mgr, 'avg_neighbors',    0.0)
    dead_arr = getattr(swarm_mgr, 'dead_mask',        None)
    n_dead   = int(np.sum(dead_arr)) if dead_arr is not None else 0
    n_total  = len(swarm_mgr.positions)
    n_alive  = n_total - n_dead
    fps      = clock.get_fps()

    # M2 metrics
    assigned     = getattr(swarm_mgr, 'assigned_tasks', None)
    n_assigned   = int(np.sum(assigned != -1)) if assigned is not None else 0
    n_completed  = getattr(swarm_mgr, 'tasks_completed', 0)
    n_tasks      = len(getattr(swarm_mgr, 'tasks', []))

    fps_color = C_TEXT_OK if fps >= 30 else (C_TEXT_WARN if fps >= 15 else (220, 60, 60))
    ts_c = C_SLOWMO if time_scale < 1.0 else (C_TEXT_WARN if time_scale > 1.0 else C_TEXT_DIM)

    # ── Left panel: sim stats ─────────────────────────────────────────────────
    left_lines = [
        ("── SIMULATION ──",            C_TEXT_DIM),
        (f"FPS:        {fps:5.1f}",     fps_color),
        (f"Alive:      {n_alive}/{n_total}", C_TEXT),
        (f"Dead:       {n_dead}",        (220,60,60) if n_dead else C_TEXT_DIM),
        (f"Method:     {method.upper()}", C_TEXT),
        (f"Avg nbrs:   {avg_nb:4.1f}",  C_TEXT),
        (f"Zoom:       {camera.zoom_pct}%", C_TEXT_DIM),
        (f"Time:       {time_scale:.1f}x", ts_c),
    ]
    pw, lh = 210, 18
    ph = len(left_lines) * lh + 14
    _panel(surface, 8, 8, pw, ph)
    y = 15
    for text, color in left_lines:
        surface.blit(font_sm.render(text, True, color), (14, y)); y += lh

    # ── Right panel: M2 task info ─────────────────────────────────────────────
    right_lines = [
        ("── MILESTONE 2 ──",           C_TEXT_DIM),
        (f"Tasks total:  {n_tasks}",    C_TEXT),
        (f"Assigned:     {n_assigned}", C_TEXT_WARN if n_assigned else C_TEXT_DIM),
        (f"Completed:    {n_completed}", C_TEXT_OK if n_completed else C_TEXT_DIM),
        ("── DISPLAY ──",               C_TEXT_DIM),
        (f"Lines [L]:    {'ON' if show_lines     else 'OFF'}", C_TEXT_DIM),
        (f"Tasks [T]:    {'ON' if show_tasks     else 'OFF'}", C_TEXT_DIM),
        (f"Form  [F]:    {'ON' if show_formation else 'OFF'}", C_TEXT_DIM),
    ]
    rw = 210
    rh = len(right_lines) * lh + 14
    _panel(surface, sw - rw - 8, 8, rw, rh)
    y = 15
    for text, color in right_lines:
        surface.blit(font_sm.render(text, True, color), (sw - rw - 2, y)); y += lh

    # ── Centre overlay ────────────────────────────────────────────────────────
    if paused:
        b = font_lg.render("PAUSED", True, C_PAUSED)
        surface.blit(b, (sw//2 - b.get_width()//2, sh//2 - b.get_height()//2))
    elif time_scale < 1.0:
        b = font_lg.render(f"SLOW  {time_scale:.1f}×", True, C_SLOWMO)
        surface.blit(b, (sw//2 - b.get_width()//2, sh//2 - b.get_height()//2))


def draw_controls_hint(surface, font_sm, sh):
    lines = [
        "SPACE pause  |  [/] time  |  L lines  |  T tasks  |  F formation  |  N/G/Q method",
        "Scroll zoom  |  Arrows/Mid-drag pan  |  R reset  |  S snapshot  |  LClick obs  |  ESC quit",
    ]
    for i, line in enumerate(lines):
        surface.blit(font_sm.render(line, True, C_TEXT_DIM),
                     (8, sh - 18 - (len(lines)-1-i)*16))


# ── Entry point ───────────────────────────────────────────────────────────────
def run_viz(swarm_mgr, env):
    """C1.1 — main loop. Called by main.py as run_viz(swarm_mgr, env)."""
    pygame.init()
    SW, SH  = env.width, env.height
    screen  = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("PDC Drone Swarm — Milestone 2  |  Boids + Decentralized Coordination")
    clock   = pygame.time.Clock()
    font_sm = pygame.font.SysFont("consolas", 13)
    font_lg = pygame.font.SysFont("consolas", 48, bold=True)

    camera         = Camera(env.width, env.height, SW, SH)
    paused         = False
    show_lines     = False   # OFF by default — big FPS win
    show_tasks     = True
    show_formation = True
    pan_active     = False
    pan_origin     = (0, 0)
    snap_count     = 0
    time_scale     = 1.0
    pair_timer     = 0

    nm = getattr(swarm_mgr, 'neighbor_mask', None)
    neighbor_pairs = pairs_from_mask(nm) if nm is not None else np.empty((0, 2), dtype=int)

    running = True
    while running:

        # ── Events ────────────────────────────────────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_SPACE:  paused = not paused
                elif k == pygame.K_l:      show_lines = not show_lines
                elif k == pygame.K_t:      show_tasks = not show_tasks
                elif k == pygame.K_f:      show_formation = not show_formation
                elif k == pygame.K_r:      camera.reset()
                elif k == pygame.K_ESCAPE: running = False
                elif k == pygame.K_s:
                    snap_count += 1
                    fname = f"swarm_snapshot_{snap_count:03d}.png"
                    pygame.image.save(screen, fname)
                    print(f"[VIZ] Snapshot: {fname}")
                elif k == pygame.K_LEFTBRACKET:
                    time_scale = round(max(TIME_SCALE_MIN, time_scale - TIME_SCALE_STEP), 2)
                elif k == pygame.K_RIGHTBRACKET:
                    time_scale = round(min(TIME_SCALE_MAX, time_scale + TIME_SCALE_STEP), 2)
                elif k == pygame.K_n:
                    if hasattr(swarm_mgr, 'set_method'): swarm_mgr.set_method('naive')
                elif k == pygame.K_g:
                    if hasattr(swarm_mgr, 'set_method'): swarm_mgr.set_method('grid')
                elif k == pygame.K_q:
                    if hasattr(swarm_mgr, 'set_method'): swarm_mgr.set_method('quadtree')

            elif event.type == pygame.MOUSEWHEEL:
                camera.zoom(1.1 if event.y > 0 else 0.9, pygame.mouse.get_pos())

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 2:
                    pan_active = True; pan_origin = event.pos
                elif event.button == 1:
                    wx, wy = camera.s2w(event.pos)
                    env._static_obstacles.append((wx, wy, NEW_OBS_RADIUS))
                    env._rebuild_obstacle_cache()
                elif event.button == 3:
                    wx, wy = camera.s2w(event.pos)
                    for idx, (ox, oy, r) in enumerate(env._static_obstacles):
                        if np.hypot(wx-ox, wy-oy) <= r:
                            env._static_obstacles.pop(idx)
                            env._rebuild_obstacle_cache()
                            break

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 2: pan_active = False

            elif event.type == pygame.MOUSEMOTION:
                if pan_active:
                    camera.pan((event.pos[0]-pan_origin[0], event.pos[1]-pan_origin[1]))
                    pan_origin = event.pos

        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:  camera.pan(( PAN_SPEED, 0))
        if keys[pygame.K_RIGHT]: camera.pan((-PAN_SPEED, 0))
        if keys[pygame.K_UP]:    camera.pan((0,  PAN_SPEED))
        if keys[pygame.K_DOWN]:  camera.pan((0, -PAN_SPEED))

        # ── Physics ───────────────────────────────────────────────────────────
        if not paused:
            swarm_mgr.update(config.dt * time_scale)
            pair_timer += 1
            if pair_timer >= PAIR_INTERVAL:
                nm = getattr(swarm_mgr, 'neighbor_mask', None)
                neighbor_pairs = (pairs_from_mask(nm) if nm is not None
                                  else fallback_pairs(swarm_mgr.positions))
                pair_timer = 0

        # ── Render ────────────────────────────────────────────────────────────
        screen.fill(C_BG)
        draw_grid(screen, env, camera, grid_step=100)
        draw_boundary(screen, env, camera)
        draw_obstacles(screen, env.obstacles, camera)

        if show_tasks:
            draw_task_zones(screen, swarm_mgr, camera)

        if show_formation:
            draw_formation(screen, swarm_mgr, camera)

        nc        = getattr(swarm_mgr, 'neighbor_counts', None)
        dead_mask = getattr(swarm_mgr, 'dead_mask', None)

        draw_neighbor_lines(screen, swarm_mgr.positions,
                            neighbor_pairs, camera, show_lines)
        draw_drones(screen, swarm_mgr.positions, swarm_mgr.velocities,
                    nc, dead_mask, camera)
        draw_hud(screen, font_sm, font_lg, clock, swarm_mgr,
                 paused, show_lines, show_tasks, show_formation,
                 time_scale, camera, SW, SH)
        draw_controls_hint(screen, font_sm, SH)

        pygame.display.flip()
        clock.tick(60)

    # Flush performance logger on exit
    logger = getattr(swarm_mgr, 'logger', None)
    if logger and hasattr(logger, 'flush'):
        logger.flush()

    pygame.quit()
    sys.exit(0)
