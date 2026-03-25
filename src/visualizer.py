"""
visualizer.py — Usman's module · PDC Project · Milestone 1
Tasks: C1.1  C1.2  C1.3  C1.4  C1.5

Entry point: run_viz(swarm_mgr=swarm_manager, env=env)

Teammates' contracts:
    env.width / env.height              — Suffiyan (A1.1)
    env.obstacles → list[(x,y,radius)] — Suffiyan (A1.2) circular
    swarm_mgr.positions  (N,2)          — Sharique (B1.1)
    swarm_mgr.velocities (N,2)          — Sharique (B1.2)
    swarm_mgr.neighbor_counts (N,)      — Ashhal   (D1.1-D1.3)
    swarm_mgr.avg_neighbors   float     — Ashhal   (D1.4)
    swarm_mgr.use_method      str       — Ashhal   (D1.2/D1.3)
    swarm_mgr.dead_mask  (N,) bool      — swarm_optimized dead drones

CONTROLS:
    SPACE       pause / resume
    [ / ]       slow down / speed up (time scale)
    L           toggle neighbor lines
    N / G / Q   switch neighbor method (naive / grid / quadtree)
    R           reset camera
    S           save snapshot
    Scroll      zoom
    Arrows      pan
    Mid-drag    pan
    LClick      add obstacle
    RClick      remove obstacle
    ESC         quit
"""

import sys
import pygame
import numpy as np
import config


# ─────────────────────────────────────────────────────────────────────────────
# VISUAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
DRONE_RADIUS   = 4
HEADING_LEN    = 12
PAN_SPEED      = 10
NEW_OBS_RADIUS = 20
PAIR_INTERVAL  = 4

TIME_SCALE_MIN  = 0.1
TIME_SCALE_MAX  = 2.0
TIME_SCALE_STEP = 0.1

C_BG            = (18,  22,  32)
C_DRONE_SOLO    = (0,  191, 255)    # blue   — isolated
C_DRONE_FLOCK   = (40, 220, 140)    # green  — has neighbors
C_DRONE_DEAD    = (220,  40,  40)   # red    — collided / dead
C_DRONE_DEAD_BD = (120,  10,  10)   # dark red border on dead drone
C_DRONE_BORDER  = (10,   80,  60)
C_HEADING       = (255, 255, 255)
C_NEIGHBOR_LINE = (40,  200, 140)
C_OBS_FILL      = (55,  28,  28)
C_OBS_BORDER    = (210,  65,  40)
C_OBS_STRIPE    = (100,  38,  28)
C_OBS_HIGHLIGHT = (240, 110,  70)
C_TEXT          = (220, 220, 220)
C_TEXT_DIM      = (110, 110, 110)
C_TEXT_WARN     = (255, 200,  50)
C_PAUSED        = (255, 200,  50)
C_SLOWMO        = (100, 200, 255)
C_BOUNDARY      = (50,   60,  80)


# ─────────────────────────────────────────────────────────────────────────────
# CAMERA
# ─────────────────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────────────────
# OBSTACLE SURFACE CACHE  (avoid per-frame Surface allocations)
# ─────────────────────────────────────────────────────────────────────────────
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
        key = (ox, oy, r, sk)
        surf = _obs_cache.get(key) or _make_obs_surf(sr)
        new[key] = surf
        pygame.draw.circle(surface, C_OBS_FILL, (sx, sy), sr)
        surface.blit(surf, (sx - sr, sy - sr))
        pygame.draw.circle(surface, C_OBS_BORDER, (sx, sy), sr, 2)
        pygame.draw.arc(surface, C_OBS_HIGHLIGHT,
                        (sx-sr, sy-sr, sr*2, sr*2), 3.14*.75, 3.14*1.5, 2)
    _obs_cache = new


# ─────────────────────────────────────────────────────────────────────────────
# NEIGHBOR PAIRS  (from swarm_mgr.neighbor_mask — Ashhal's data)
# ─────────────────────────────────────────────────────────────────────────────
def pairs_from_mask(mask):
    p = np.argwhere(mask)
    return p[p[:, 0] < p[:, 1]]

def fallback_pairs(positions):
    diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    p = np.argwhere(dist < config.perception_radius)
    return p[p[:, 0] < p[:, 1]]


# ─────────────────────────────────────────────────────────────────────────────
# DRAW FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def draw_boundary(surface, env, camera):
    x1, y1 = camera.ws((0, 0))
    x2, y2 = camera.ws((env.width, env.height))
    pygame.draw.rect(surface, C_BOUNDARY, (x1, y1, x2-x1, y2-y1), 1)


def draw_neighbor_lines(surface, positions, pairs, camera, show):
    if not show or len(pairs) == 0: return
    sw, sh = surface.get_size()
    for i, j in pairs:
        sx1, sy1 = camera.ws(positions[i])
        sx2, sy2 = camera.ws(positions[j])
        if (max(sx1,sx2)<-50 or min(sx1,sx2)>sw+50 or
                max(sy1,sy2)<-50 or min(sy1,sy2)>sh+50): continue
        pygame.draw.line(surface, C_NEIGHBOR_LINE, (sx1,sy1), (sx2,sy2), 1)


def draw_drones(surface, positions, velocities, neighbor_counts, dead_mask, camera):
    """C1.2 — draw each drone.

    Color coding:
      RED   — dead (collided with another drone)
      GREEN — alive, has neighbors (flocking)
      BLUE  — alive, isolated

    Dead drones show an X cross instead of a heading line to make their
    state immediately obvious even at small zoom levels.
    """
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
            # Red filled circle
            pygame.draw.circle(surface, C_DRONE_DEAD, (sx, sy), max(r, 3))
            pygame.draw.circle(surface, C_DRONE_DEAD_BD, (sx, sy), max(r, 3), 1)
            # X cross to signal dead
            cx = max(r - 1, 2)
            pygame.draw.line(surface, C_DRONE_DEAD_BD,
                             (sx-cx, sy-cx), (sx+cx, sy+cx), 2)
            pygame.draw.line(surface, C_DRONE_DEAD_BD,
                             (sx+cx, sy-cx), (sx-cx, sy+cx), 2)
        else:
            color = (C_DRONE_FLOCK if has_nc and neighbor_counts[i] > 0
                     else C_DRONE_SOLO)
            pygame.draw.circle(surface, color, (sx, sy), max(r, 2))
            if r > 2:
                pygame.draw.circle(surface, C_DRONE_BORDER, (sx, sy), r, 1)

            speed = np.linalg.norm(velocities[i])
            if speed > 0.01:
                nv = velocities[i] / speed
                pygame.draw.line(surface, C_HEADING, (sx, sy),
                                 (int(sx + nv[0]*head_len),
                                  int(sy + nv[1]*head_len)), 1)


def draw_hud(surface, font_sm, font_lg, clock, swarm_mgr,
             paused, show_lines, time_scale, camera, sw, sh):
    """C1.4 — stats overlay.

    Rows:
      FPS            render performance
      Drones alive   live count (decreases as drones collide)
      Dead           collision fatalities
      Method         N/G/Q active algorithm
      Avg neighbors  spatial indexing payoff metric
      Zoom           camera scale
      Neighbor viz   L key toggle
      Time scale     [ / ] multiplier
    """
    method   = getattr(swarm_mgr, 'use_method',    'N/A')
    avg_nb   = getattr(swarm_mgr, 'avg_neighbors',  0.0)
    dead_arr = getattr(swarm_mgr, 'dead_mask',       None)
    n_dead   = int(np.sum(dead_arr)) if dead_arr is not None else 0
    n_alive  = len(swarm_mgr.positions) - n_dead
    fps      = clock.get_fps()

    ts_c = C_SLOWMO if time_scale < 1.0 else (C_TEXT_WARN if time_scale > 1.0 else C_TEXT_DIM)

    lines = [
        (f"FPS:           {fps:5.1f}",                              C_TEXT),
        (f"Drones alive:  {n_alive}",                               C_TEXT),
        (f"Method:        {method.upper()}",                        C_TEXT),
        (f"Avg neighbors: {avg_nb:5.1f}",                           C_TEXT),
        (f"Zoom:          {camera.zoom_pct}%",                      C_TEXT_DIM),
        (f"Neighbor viz:  {'ON' if show_lines else 'OFF'}",         C_TEXT_DIM),
        (f"Time scale:    {time_scale:.1f}x",                       ts_c),
    ]

    pw = 240; ph = len(lines)*20+14
    p  = pygame.Surface((pw, ph), pygame.SRCALPHA)
    p.fill((0,0,0,170)); surface.blit(p, (8,8))

    y = 15
    for text, color in lines:
        surface.blit(font_sm.render(text, True, color), (14, y))
        y += 20

    # Dead drone counter — separate prominent box, always visible
    dead_bg_color = (100, 10, 10, 200) if n_dead > 0 else (0, 0, 0, 120)
    dead_txt_color = (255, 80, 80) if n_dead > 0 else C_TEXT_DIM
    dead_label = f"DEAD: {n_dead}"
    db = pygame.Surface((120, 28), pygame.SRCALPHA)
    db.fill(dead_bg_color)
    surface.blit(db, (8, ph + 14))
    surface.blit(font_sm.render(dead_label, True, dead_txt_color), (14, ph + 20))

    if paused:
        b = font_lg.render("PAUSED", True, C_PAUSED)
        surface.blit(b, (sw//2-b.get_width()//2, sh//2-b.get_height()//2))
    elif time_scale < 1.0:
        b = font_lg.render(f"SLOW  {time_scale:.1f}\u00d7", True, C_SLOWMO)
        surface.blit(b, (sw//2-b.get_width()//2, sh//2-b.get_height()//2))


def draw_controls_hint(surface, font_sm, sh):
    hint = ("SPACE pause  |  [ ] slow/fast  |  L lines  |  N/G/Q method  |  "
            "Scroll zoom  |  Arrows/Mid-drag pan  |  R reset  |  S snapshot  |  "
            "LClick add obs  |  RClick remove obs")
    surface.blit(font_sm.render(hint, True, C_TEXT_DIM), (8, sh-18))


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def run_viz(swarm_mgr, env, logger=None):
    """C1.1 — main loop. Called by main.py as run_viz(swarm_mgr, env)."""
    pygame.init()
    SW, SH = env.width, env.height
    screen  = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("PDC Drone Swarm — Usman (C1.1–C1.5) · M1")
    clock   = pygame.time.Clock()
    font_sm = pygame.font.SysFont("monospace", 13)
    font_lg = pygame.font.SysFont("monospace", 48, bold=True)

    camera     = Camera(env.width, env.height, SW, SH)
    paused     = False
    show_lines = True
    pan_active = False
    pan_origin = (0, 0)
    snap_count = 0
    time_scale = 1.0
    pair_timer = 0

    nm = getattr(swarm_mgr, 'neighbor_mask', None)
    neighbor_pairs = pairs_from_mask(nm) if nm is not None else fallback_pairs(swarm_mgr.positions)

    running = True
    while running:

        # 1. Start the frame timer
        if logger:
            logger.start_frame()

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                if   k == pygame.K_SPACE:   paused = not paused
                elif k == pygame.K_l:       show_lines = not show_lines
                elif k == pygame.K_r:       camera.reset()
                elif k == pygame.K_ESCAPE:  running = False
                elif k == pygame.K_s:
                    snap_count += 1
                    fname = f"swarm_snapshot_{snap_count:03d}.png"
                    pygame.image.save(screen, fname)
                    print(f"[VIZ] Snapshot: {fname}")
                elif k == pygame.K_LEFTBRACKET:
                    time_scale = round(max(TIME_SCALE_MIN, time_scale-TIME_SCALE_STEP), 2)
                elif k == pygame.K_RIGHTBRACKET:
                    time_scale = round(min(TIME_SCALE_MAX, time_scale+TIME_SCALE_STEP), 2)
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
                    env.obstacles.append((wx, wy, NEW_OBS_RADIUS))
                elif event.button == 3:
                    wx, wy = camera.s2w(event.pos)
                    for idx, (ox,oy,r) in enumerate(env.obstacles):
                        if np.hypot(wx-ox, wy-oy) <= r:
                            env.obstacles.pop(idx); break

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

        # Physics
        if not paused:
            swarm_mgr.update(config.dt * time_scale)
            pair_timer += 1
            if pair_timer >= PAIR_INTERVAL:
                nm = getattr(swarm_mgr, 'neighbor_mask', None)
                neighbor_pairs = (pairs_from_mask(nm) if nm is not None
                                  else fallback_pairs(swarm_mgr.positions))
                pair_timer = 0

        # Render
        screen.fill(C_BG)
        draw_boundary(screen, env, camera)
        draw_obstacles(screen, env.obstacles, camera)

        nc        = getattr(swarm_mgr, 'neighbor_counts', None)
        dead_mask = getattr(swarm_mgr, 'dead_mask', None)

        draw_neighbor_lines(screen, swarm_mgr.positions,
                            neighbor_pairs, camera, show_lines)
        draw_drones(screen, swarm_mgr.positions, swarm_mgr.velocities,
                    nc, dead_mask, camera)
        draw_hud(screen, font_sm, font_lg, clock, swarm_mgr,
                 paused, show_lines, time_scale, camera, SW, SH)
        draw_controls_hint(screen, font_sm, SH)

        pygame.display.flip()
        clock.tick(60)

        if logger:
            # Pass the swarm_mgr so the logger can read drone counts/methods
            logger.end_frame(swarm_mgr)

    pygame.quit()
    sys.exit(0)
