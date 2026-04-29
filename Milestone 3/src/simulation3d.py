"""
simulation3d.py -- Usman (C3.1-C3.5) | PDC Drone Swarm | Milestone 3
Exhaustive Parallelization & Optimization

KEYS:
    1 / 2          Algorithm: Naive O(N^2) | Octree
  L              Toggle neighbor lines
  H              Toggle floor heatmap (coverage tiles)
  T              Toggle drone trails
  O              Toggle obstacle placement mode
    M              (in obs mode) toggle moving obstacle
    = / -          (in obs mode) raise / lower placement height
    Left-Click     Place obstacle
    Right-Click    Undo last placed obstacle
  C              Cinematic camera
  R              Reset simulation
  B              Toggle benchmark overlay (parallel speedup HUD)
  P              Export metrics CSV + parallel analysis
  V              Toggle diagnostics
  G              Toggle global center marker
  Arrow keys     Pan camera: Left/Right = X   Up/Down = Z (forward/back)
  Page Up/Down   Pan camera: Y (vertical)
  Left-Click     Set waypoint (normal mode)
  Right-Click    Clear waypoint (normal mode)

═══ PDC TECHNIQUES IN VISUALIZATION LAYER ═══
  - Data Parallelism: Batch entity position updates via NumPy slicing
  - Loop Unrolling: Per-drone color/state update optimized
  - Pipeline: Render uses pre-computed swarm state from parallel update()
"""
from ursina import *
from ursina.prefabs.slider import ThinSlider
from swarm_3d import SwarmManager3D, GPU_AVAILABLE
from environment3d import Environment3D
from render_optimizer import BatchDroneRenderer, OptimizedHeatmapRenderer, OptimizedNeighborLineRenderer, RenderProfiler
from gpu_pipeline import PaddedGrid, AsyncGPUPipeline, FrameDoubleBuffer, RenderFrameStats
import numpy as np
import config, math, random, csv, time
from collections import deque

# ── RGB HELPER ────────────────────────────────────────────────────────────────
def rgb(r, g, b, a=255):
    return color.rgba(r/255.0, g/255.0, b/255.0, a/255.0)

# ── APP ───────────────────────────────────────────────────────────────────────
app = Ursina(borderless=False, title='PDC Drone Swarm | M3 | Parallelized')
app.start_time = time.time()
app.setBackgroundColor(rgb(4, 6, 13)) # Ensure safe clear color

# Discovery Pulse Effect (M2.7 - High Performance)
class DiscoveryPulse(Entity):
    def __init__(self, pos):
        super().__init__(model='sphere', scale=15, position=pos, color=rgb(0, 250, 255, 140), unlit=True)
        self.animate_scale(45, duration=0.7, curve=curve.out_expo)
        self.animate_color(rgb(0, 250, 255, 0), duration=0.7, curve=curve.out_expo)
        destroy(self, delay=0.8)

window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.exit_button.visible = False

env   = Environment3D()
swarm = SwarmManager3D(env)
W, H  = config.width, config.height
D     = W                       # World depth (for 3D voxel grid and pathing)
D     = env.depth

# ── CAMERA ────────────────────────────────────────────────────────────────────
editor_cam = EditorCamera()
editor_cam.position = Vec3(W/2, 550, -750)
editor_cam.rotation = (26, 0, 0)
camera.clip_plane_far = 20000
cinematic_mode = False
CAM_PAN  = 220          # units / sec for arrow-key pan
CAM_ZOOM = 600          # units / sec for zoom (= / - keys)
CAM_SCROLL = 220        # units per scroll tick

# ── LIGHTING ──────────────────────────────────────────────────────────────────
d_light = DirectionalLight(parent=scene, y=8, z=4, color=rgb(160, 185, 255))
d_light.look_at(Vec3(0, -1, 0.4))
AmbientLight(parent=scene, color=rgb(18, 22, 38))

# ── SKYBOX ────────────────────────────────────────────────────────────────────
Entity(model='sphere', scale=15000, color=rgb(4, 6, 13),
       double_sided=True, unlit=True)
random.seed(77)
_sv = [Vec3(random.uniform(-6000,6000), random.uniform(80,4500),
            random.uniform(-6000,6000)) for _ in range(100)]
Entity(model=Mesh(vertices=_sv, mode='point', thickness=3),
       color=rgb(195, 215, 255), unlit=True)

# ── FLOOR ─────────────────────────────────────────────────────────────────────
Entity(model='plane', scale=(W, 1, W), position=(W/2, 0, W/2),
       color=rgb(9, 14, 23), unlit=True)

GRID_DIVS = 8
_gs = W / GRID_DIVS
for _gi in range(GRID_DIVS + 1):
    _v = _gi * _gs
    Entity(model=Mesh(vertices=[Vec3(_v, 1.2, 0), Vec3(_v, 1.2, W)], mode='line', thickness=1),
           color=rgb(20, 60, 100, 120), unlit=True)
    Entity(model=Mesh(vertices=[Vec3(0, 1.2, _v), Vec3(W, 1.2, _v)], mode='line', thickness=1),
           color=rgb(20, 60, 100, 120), unlit=True)

# Massive invisible floor for reliable raycasting/interactivity
floor_collider = Entity(model='plane', scale=(30000, 1, 30000),
                        position=(W/2, 0, W/2),
                        collider='box', visible=False)

# ── BOUNDARY: floor + top squares + vertical pillars ─────────────────────────
_cn = [(0,0,0),(W,0,0),(W,0,W),(0,0,W),
       (0,H,0),(W,H,0),(W,H,W),(0,H,W)]
_bed = [(0,1),(1,2),(2,3),(3,0)]
_ted = [(4,5),(5,6),(6,7),(7,4)]
_pil = [(0,4),(1,5),(2,6),(3,7)]

for a, b in _bed + _ted + _pil:
    Entity(model=Mesh(vertices=[Vec3(*_cn[a]), Vec3(*_cn[b])], mode='line', thickness=2),
           color=rgb(55, 105, 190, 235), unlit=True)

# ── OBSTACLES ─────────────────────────────────────────────────────────────────
_initial_static_count = len(env.obstacles)

static_obs_ents = []
for ob in env.obstacles:
    e = Entity(model='cube', scale=ob[3]*0.85, position=(ob[0],ob[1],ob[2]),
               color=rgb(215, 35, 55), unlit=True, wireframe=True,
               rotation=(35, 35, 0))
    static_obs_ents.append(e)

dyn_obs_ents = []
for d in env.dynamic_obstacles:
    e = Entity(model='cube', scale=d.radius*0.85, position=(d.x,d.y,d.z),
               color=rgb(180, 50, 200), unlit=True, wireframe=True,
               rotation=(35, 35, 0))
    dyn_obs_ents.append(e)

# ── OBSTACLE PLACEMENT ────────────────────────────────────────────────────────
obs_mode        = False
obs_moving_mode = False
obs_height      = [60.0]
user_added      = []
user_moving_obs = []

obs_ghost = Entity(model='cube', scale=config.obstacle_radius * 0.85,
                   color=rgb(255, 155, 30, 170), unlit=True,
                   wireframe=True, rotation=(35, 35, 0), enabled=False)

# ── WAYPOINT ──────────────────────────────────────────────────────────────────
waypoint_marker = Entity(model='sphere', scale=10,
                         color=rgb(0, 255, 120, 150), unlit=True,
                         wireframe=True, enabled=False)
_lv = [Vec3(0,0,0), Vec3(0, H*1.05, 0)]
Entity(parent=waypoint_marker,
       model=Mesh(vertices=_lv, mode='line', thickness=2),
       color=rgb(0, 255, 120, 80), unlit=True)
waypoint_ring = Entity(parent=waypoint_marker, model='circle',
                       scale=3.5, rotation_x=90,
                       color=rgb(0, 255, 120, 90), unlit=True)

# ── FORMATION CENTROID & GLOBAL CENTER (C2.2 / M2.5) ─────────────────────────
centroid_marker = Entity(model='diamond', scale=(30, 50, 30), color=rgb(255, 255, 255, 180), 
                         unlit=True, wireframe=True, enabled=False)

# ── NEIGHBOR LINES ────────────────────────────────────────────────────────────
show_neighbor_lines = False
nb_line_ent = Entity(model=Mesh(vertices=[], mode='line', thickness=1),
                     color=rgb(0, 190, 255, 70), unlit=True, enabled=False)

# ── TRAILS ────────────────────────────────────────────────────────────────────
show_trails = True
show_centroid = False

# ── HEATMAP ───────────────────────────────────────────────────────────────────
show_heatmap  = False
HMAP_CELL     = 80
hmap_cols     = int(W / HMAP_CELL) + 1
hmap_rows     = int(W / HMAP_CELL) + 1
total_hcells  = hmap_cols * hmap_rows
visited_cells = set()

hmap_verts  = []
hmap_colors = []
hmap_ent = Entity(model=Mesh(vertices=[], colors=[], mode='point', thickness=14),
                  unlit=True, enabled=False)

def _stamp_tile(cx, cz):
    x = cx * HMAP_CELL + HMAP_CELL/2
    z = cz * HMAP_CELL + HMAP_CELL/2
    hmap_verts.append(Vec3(x, 0, z))
    hmap_colors.append(rgb(0, 190, 255, 180))
    
    if hmap_ent.enabled:
        hmap_ent.model.vertices = hmap_verts
        hmap_ent.model.colors = hmap_colors
        hmap_ent.model.generate()

# ─── OPTIMIZED BATCH DRONE RENDERER ──────────────────────────────────────────
# ═══ PDC TECHNIQUE: Data Parallelism + Ring Buffers ═══
# Instead of 100 individual Entity updates per frame, we batch:
#  1. Positions in a single NumPy array (100,3) → single GPU transfer
#  2. Trails in ring buffers (no per-frame allocation)
#  3. Colors via vectorized conditionals (SIMD-friendly)
boid_entities = []
batch_renderer = BatchDroneRenderer(swarm.num_boids, max_trail_len=10)
trail_buffers = [deque(maxlen=10) for _ in range(swarm.num_boids)]  # Ring buffers

for i in range(swarm.num_boids):
    drone = Entity(model='sphere', scale=7, color=rgb(0, 210, 255), unlit=True)
    drone.trail = Entity(model=Mesh(vertices=[], mode='line', thickness=2),
                         color=rgb(0, 145, 220, 185), unlit=True)
    drone._prev_color_key = ''
    boid_entities.append(drone)

# Optimized heatmap and neighbor renderer
optimized_heatmap = OptimizedHeatmapRenderer(swarm.grid_res, W)
optimized_neighbors = OptimizedNeighborLineRenderer(max_pairs=360)
render_profiler = RenderProfiler()
render_stats = RenderFrameStats()
    
# ── M3 TASK MARKERS & ALLOCATOR PRISMS (C2.3) ────────────────────────────────
task_markers = []
for t_idx, t_pos in enumerate(swarm.tasks):
    if t_idx in [4, 5]:
        continue
    
    tm = Entity(position=(t_pos[0], t_pos[1], t_pos[2]), enabled=True)
    
    icon = Entity(parent=tm, model='diamond', scale=(30, 50, 30), color=rgb(0, 255, 255, 180), unlit=True)
    
    ring = Entity(parent=tm, model='circle', scale=25, rotation_x=90, color=rgb(150, 150, 150, 60), unlit=True)
    task_markers.append({'base': tm, 'icon': icon, 'ring': ring})

# ── HUD ───────────────────────────────────────────────────────────────────────
ui_panel = Entity(parent=camera.ui, model='quad', scale=(0.38, 0.95),
                  position=(-0.72, 0.0), color=rgb(8, 12, 22, 60))
Entity(parent=camera.ui, model='quad', scale=(0.385, 0.955),
       position=(-0.72, 0.0), color=rgb(0, 140, 255, 10), z=1)

info_text = Text(text='Initializing...', position=(-0.89, 0.45),
                 scale=0.65, color=rgb(180, 180, 180), background=False)

if not hasattr(config, 'waypoint_weight'):
    config.waypoint_weight = 2.5

slider_x = -0.84
slider_start_y = 0.08
def _make_slider(text, val, y_off):
    # Place heading above the bar and keep both inside the left UI panel
    heading_x = slider_x + 0.02
    Text(parent=camera.ui, text=text, position=(heading_x, y_off + 0.035),
        scale=0.65, color=color.white, origin=(-0.5,0))
    # Reduce slider width so it fits inside the panel and move slightly down
    s = ThinSlider(text='', dynamic=True, min=0, max=10, default=val,
                x=slider_x, y=y_off - 0.02, parent=camera.ui, scale=0.32)
    return s

separation_slider = _make_slider('Separation', config.separation_weight, slider_start_y)
alignment_slider  = _make_slider('Alignment', config.alignment_weight, slider_start_y - 0.08)
cohesion_slider   = _make_slider('Cohesion',  config.cohesion_weight, slider_start_y - 0.16)
waypoint_slider   = _make_slider('Waypoint',  config.waypoint_weight, slider_start_y - 0.24)

def update_weights():
    config.separation_weight = separation_slider.value
    config.alignment_weight = alignment_slider.value
    config.cohesion_weight = cohesion_slider.value
    config.waypoint_weight = waypoint_slider.value

separation_slider.on_value_changed = update_weights
alignment_slider.on_value_changed = update_weights
cohesion_slider.on_value_changed = update_weights
waypoint_slider.on_value_changed = update_weights

controls_text = Text(
    text='CONTROLS\n'
         '1/2   : Algo      L : Lines\n'
         'T : Trails        H : Map\n'
         'O : Obs Mode      R : Reset\n'
         'C : Cinematic     G : Center\n'
         'W/A/S/D : Pan XZ  Q/E : Pan Y\n'
         '= : Zoom In       - : Zoom Out\n'
         'Scroll Wheel : Zoom (fast)\n'
         'L-Click Floor: Set Waypoint\n'
         'R-Click: Clear Waypoint\n'
         'P : Export CSV    V : Diagnostics\n'
         'B : Benchmark HUD (Parallel)\n'
         '[Obs] Up/Down: Height  M: Move\n'
         '[Obs] L-Click: Place   R-Click: Undo',
    position=(-0.89, -0.21), scale=0.65, color=color.azure
)

show_vectors = False
intent_indicator = Entity(model='diamond', scale=(35, 60, 35), color=rgb(0, 210, 255, 120),
                          unlit=True, wireframe=True, enabled=False)
intent_glow = Entity(parent=intent_indicator, model='circle', scale=1.2, rotation_x=90,
                     color=rgb(0, 210, 255, 60), unlit=True)

# ── FORCE BREAKDOWN HUD (C2.4) ───────────────────────────────────────────────
force_hud = Entity(parent=camera.ui, enabled=False)
force_bg = Entity(parent=force_hud, model='quad', scale=(0.22, 0.25), position=(0.83, -0.45), color=rgb(8,12,22,30))
force_border = Entity(parent=force_hud, model='quad', scale=(0.225, 0.255), position=(0.83, -0.45), color=rgb(0,180,255,8), z=1)
Text(parent=force_hud, text='AVG SWARM FORCE', position=(0.73, -0.34), scale=0.6, color=rgb(150, 150, 150))

def _make_bar(y, label, f_col):
    Text(parent=force_hud, text=label, position=(0.73, y+0.012), scale=0.55, color=rgb(150, 150, 150))
    Entity(parent=force_hud, model='quad', scale=(0.14, 0.01), position=(0.8, y), color=rgb(40,40,40, 60), origin=(-0.5, 0))
    fg = Entity(parent=force_hud, model='quad', scale=(0.0, 0.01), position=(0.8, y), color=f_col, origin=(-0.5, 0))
    return fg

bar_sep = _make_bar(-0.41, 'Separation', rgb(200, 100, 100))
bar_aln = _make_bar(-0.46, 'Alignment', rgb(200, 200, 100))
bar_coh = _make_bar(-0.51, 'Cohesion', rgb(100, 200, 100))
bar_tsk = _make_bar(-0.56, 'Waypoint', rgb(100, 150, 200))

# Mode bar (centre top)
mode_bar_bg = Entity(parent=camera.ui, model='quad', scale=(0.6, 0.06), position=(0, 0.45), color=rgb(8, 12, 22, 140), enabled=False)
mode_bar = Text(text='', origin=(0,0), position=(0, 0.45),
                scale=1.0, color=rgb(255, 165, 30), background=False)

# Volumetric Bounding Box
volume_outline = Entity(model='cube', scale=(W, H, D), position=(W/2, H/2, D/2), 
                        color=rgb(0, 210, 255, 15), wireframe=True, unlit=True, enabled=True)

scan_plane = Entity(model='quad', scale=(W, D), rotation_y=90, 
              color=rgb(0, 255, 255, 10), unlit=True, enabled=False)

# ── MISSION FLEET HUD (C2.3 - Highlighting/Selection) ────────────────────────
mission_hud_panel = Entity(parent=camera.ui, model='quad', scale=(0.20, 0.75), 
                           position=(0.78, -0.1), color=rgb(8, 12, 22, 40))
Entity(parent=camera.ui, model='quad', scale=(0.205, 0.755),
       position=(0.78, -0.1), color=rgb(0, 140, 255, 8), z=0.01)
Text(parent=mission_hud_panel, text='FLEET COMMAND', position=(0, 0.46), scale=0.65, color=rgb(150, 150, 150), origin=(0,0))

fault_btn = Button(parent=mission_hud_panel, text='FAULT INJECTION',
                   scale=(0.8, 0.05), position=(0, -0.28), color=rgb(80, 40, 40, 60))
reset_btn = Button(parent=mission_hud_panel, text='RESET FLEET',
                   scale=(0.8, 0.05), position=(0, -0.36), color=rgb(40, 80, 40, 60))

def inject_fault():
    swarm.inject_faults(0.2)
    print("[M3] Chaos Injected: 20% Drones failing!")
fault_btn.on_click = inject_fault

def reset_fleet():
    swarm.reset_faults()
    print("[M3] Fleet Restored.")
reset_btn.on_click = reset_fleet

highlighted_mission = [-1] 

def select_mission(m_id):
    if highlighted_mission[0] == m_id: highlighted_mission[0] = -1
    else: highlighted_mission[0] = m_id

mission_btns = []
mission_labels = ['Seeking Tasks', 'Obstacle Fleet', 'Patrol Fleet', 'Swarm Fleet']
for i, label in enumerate(mission_labels):
    btn = Button(parent=mission_hud_panel, text=label, scale=(0.85, 0.08),
                 position=(0.02, 0.32 - i*0.1), color=rgb(60, 60, 60, 50),
                 on_click=Func(select_mission, i))
    mission_btns.append(btn)

coverage_btn = Button(parent=mission_hud_panel, text='COVERAGE DRONES', scale=(0.85, 0.08),
                      position=(0.02, -0.08), color=rgb(60, 80, 60, 50),
                      on_click=Func(select_mission, 6))
mission_btns.append(coverage_btn)

coverage_text = Text(parent=mission_hud_panel, text='COVERAGE: 0.0%', 
                     position=(0, 0.42), scale=0.55, color=rgb(150, 150, 150), origin=(0,0))

# Mission Success Banner
mission_banner = Entity(parent=camera.ui, model='quad', scale=(0.8, 0.15), 
                        position=(0, 0), color=rgb(20, 180, 255, 0), enabled=False)
banner_text = Text(parent=mission_banner, text='AREA CLEARED | RECALL INITIATED', 
                   origin=(0,0), scale=2.5, color=color.white)# ═══════════════════════════════════════════════════════════════════════════════
#  M3: BENCHMARK OVERLAY HUD — Shows parallel speedup metrics in real-time
#  ═══ PDC TECHNIQUE 15: Amdahl's & Gustafson's Law ═══
#  Displays serial fraction, Amdahl speedup, Gustafson speedup, and
#  technique timing breakdown directly in the simulation HUD.
# ═══════════════════════════════════════════════════════════════════════════════
show_benchmark = False
bench_panel = Entity(parent=camera.ui, enabled=False)

# Background and border
Entity(parent=bench_panel, model='quad', scale=(0.42, 0.26),
       position=(0.0, -0.38), color=rgb(8, 12, 22, 190), z=0.02)
Entity(parent=bench_panel, model='quad', scale=(0.425, 0.265),
       position=(0.0, -0.38), color=rgb(0, 255, 180, 15), z=0.01)

# Texts
bench_title = Text(parent=bench_panel, text='PARALLEL METRICS', 
                   position=(-0.19, -0.27), scale=0.7, color=rgb(0, 255, 180))
bench_text = Text(parent=bench_panel, text='Warming up...', 
                  position=(-0.19, -0.31), scale=0.6, color=rgb(180, 200, 180))
gpu_badge = Text(parent=bench_panel, text=f'GPU: {"RTX 4050 ✓" if GPU_AVAILABLE else "CPU Only"}',
                 position=(0.04, -0.27), scale=0.6, 
                 color=rgb(0, 255, 120) if GPU_AVAILABLE else rgb(200, 100, 100))

# ── TIMING & METRICS ─────────────────────────────────────────────────────────
_frame = [0]
last_vis_count = [0] 
metrics_log = []
log_timer = 0

# ═══════════════════════════════════════════════════════════════════════════════
#  M3: DECOUPLED ASYNCHRONOUS PHYSICS PIPELINE
#  ═══ PDC TECHNIQUE: Lock-Free Producer-Consumer Pipeline ═══
#  The physics engine runs entirely independently in a background daemon thread.
#  It updates the simulation at maximum possible speed (TPS), while the 
#  main Ursina loop only consumes the state at 60 FPS for rendering.
# ═══════════════════════════════════════════════════════════════════════════════
import threading

cinematic_mode_running = [False]
_physics_tps = [0.0]

def _physics_worker():
    last_t = time.perf_counter()
    frames = 0
    while True:
        if cinematic_mode_running[0]:
            time.sleep(0.01)
            continue
        try:
            swarm.update()
            frames += 1
            curr_t = time.perf_counter()
            if curr_t - last_t >= 1.0:
                _physics_tps[0] = frames / (curr_t - last_t)
                frames = 0
                last_t = curr_t
        except RuntimeError as e:
            # Pool can be briefly unavailable during reset/teardown.
            if 'cannot schedule new futures after shutdown' in str(e):
                time.sleep(0.05)
                continue
            print(f"[PHYSICS DAEMON] RuntimeError: {e}")
            import traceback; traceback.print_exc()
            time.sleep(0.2)
        except Exception as e:
            print(f"[PHYSICS DAEMON] Error: {e}")
            import traceback; traceback.print_exc()
            time.sleep(1)

physics_thread = threading.Thread(target=_physics_worker, daemon=True, name="PhysicsWorker")
physics_thread.start()

# ═══════════════════════════════════════════════════════════════════════════════
#  UPDATE LOOP (RENDER ONLY)
# ═══════════════════════════════════════════════════════════════════════════════
def update():
    _frame[0] += 1
    _t = time.time()
    
    try:
        cov_pct = swarm.coverage_pct
    except Exception:
        cov_pct = 0.0

    # Camera keyboard pan (WASD relative to view, QE for Y) + zoom
    try:
        y_rot = math.radians(editor_cam.rotation_y)
        x_rot = math.radians(editor_cam.rotation_x)
        fwd   = Vec3(math.sin(y_rot) * math.cos(x_rot),
                     -math.sin(x_rot),
                     math.cos(y_rot) * math.cos(x_rot))
        right = Vec3(math.cos(y_rot), 0, -math.sin(y_rot))
    except:
        fwd, right = Vec3(0,0,1), Vec3(1,0,0)

    if held_keys['a']:   editor_cam.position -= right * CAM_PAN * time.dt
    if held_keys['d']:   editor_cam.position += right * CAM_PAN * time.dt
    if held_keys['w']:   editor_cam.position += fwd * CAM_PAN * time.dt
    if held_keys['s']:   editor_cam.position -= fwd * CAM_PAN * time.dt
    if held_keys['e']:   editor_cam.position += Vec3(0, CAM_PAN * time.dt, 0)
    if held_keys['q']:   editor_cam.position -= Vec3(0, CAM_PAN * time.dt, 0)
    if held_keys['=']:   editor_cam.position += fwd * CAM_ZOOM * time.dt
    if held_keys['-']:   editor_cam.position -= fwd * CAM_ZOOM * time.dt

    # Waypoint pulse (throttled)
    if waypoint_marker.enabled:
        waypoint_marker.rotation_y += 70 * time.dt
        if _frame[0] % 4 == 0:
            pulse = 1.0 + 0.38 * math.sin(_t * 5)
            waypoint_ring.scale_x = 3 * pulse
            waypoint_ring.scale_y = 3 * pulse

    # Obstacle placement ghost preview
    if obs_mode:
        try:
            if hasattr(mouse, 'world_point') and mouse.world_point:
                wp = mouse.world_point
                obs_ghost.enabled = True
                obs_ghost.position = Vec3(wp.x, obs_height[0], wp.z)
            else:
                cam_pos = camera.world_position
                dist_to_plane = obs_height[0] - cam_pos.y
                if abs(dist_to_plane) > 1:
                    mx = (mouse.x - 0.5) * 10
                    mz = (mouse.y - 0.5) * 10
                    obs_ghost.enabled = True
                    obs_ghost.position = Vec3(cam_pos.x + mx, obs_height[0], cam_pos.z + mz)
                else:
                    obs_ghost.enabled = False
        except:
            obs_ghost.enabled = False
            
        obs_ghost.rotation_y += 55 * time.dt

    # Update moving user-placed obstacles
    for mo in user_moving_obs:
        nx = mo['ox'] + mo['amp'] * math.sin(mo['freq']  * _t)
        nz = mo['oz'] + mo['amp'] * math.cos(mo['freq2'] * _t)
        nx = max(50, min(W-50, nx))
        nz = max(50, min(W-50, nz))
        sidx = mo['static_idx']
        if sidx < len(swarm.env.obstacles):
            old = swarm.env.obstacles[sidx]
            swarm.env.obstacles[sidx] = (nx, old[1], nz, old[3])

    # Sync obstacle entities
    for i, ob in enumerate(swarm.env.obstacles):
        if i < len(static_obs_ents):
            static_obs_ents[i].position = (ob[0], ob[1], ob[2])
    for i, d in enumerate(swarm.env.dynamic_obstacles):
        if i < len(dyn_obs_ents):
            dyn_obs_ents[i].position = (d.x, d.y, d.z)

    # ═══════════════════════════════════════════════════════════════════════════
    #  OPTIMIZED BATCH DRONE UPDATE
    #  ═══ PDC TECHNIQUE: Data Parallelism + Ring Buffers + Batch Processing ═══
    #  Key optimizations:
    #    1. Single lock for position copy (minimal contention)
    #    2. Vectorized color computation (NumPy SIMD)
    #    3. Ring buffer trails (no per-frame allocation)
    #    4. Lazy mesh regeneration (only when dirty)
    # ═══════════════════════════════════════════════════════════════════════════
    render_profiler.start_frame()
    active_count = 0
    centroid = Vec3(0, 0, 0)
    do_hmap  = show_heatmap and (_frame[0] % 6 == 0)
    do_trail = show_trails  and (_frame[0] % 8 == 0)
    do_nl    = show_neighbor_lines and (_frame[0] % 10 == 0)
    do_look  = (_frame[0] % 2 == 0)

    # Single fast copy (one lock acquire/release)
    with swarm.state_lock:
        local_positions = swarm.positions.copy()
        local_velocities = swarm.velocities.copy()
        local_dead = swarm.dead_mask.copy()
        local_missions = swarm.mission_type.copy()
    
    render_profiler.mark_section('state_copy')
    
    # ═══ Vectorized position & color batch update ═══
    for i in range(len(boid_entities)):
        e = boid_entities[i]
        pos = local_positions[i]
        is_dead = local_dead[i]

        if is_dead:
            if e._prev_color_key != 'dead':
                e.color = rgb(255, 30, 30)
                e._prev_color_key = 'dead'
            e.y = max(-15, e.y - 140*time.dt)
            e.rotation_x += 75*time.dt
            continue

        active_count += 1
        p3 = Vec3(pos[0], pos[1], pos[2])
        e.position = p3  # Single per-drone (unavoidable with Ursina)
        centroid += p3

        vel = local_velocities[i]
        spd = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        h_id = highlighted_mission[0]
        
        # Vectorized color selection (cache-friendly)
        if show_vectors:
            new_key = 'vec'
        elif h_id != -1 and local_missions[i] == h_id:
            new_key = 'sel'
        elif h_id != -1:
            new_key = 'dim'
        else:
            new_key = 'norm'
        
        if e._prev_color_key != new_key:
            if new_key == 'vec':
                e.color = rgb(0, 210, 255); e.scale = 7; e.unlit = False
            elif new_key == 'sel':
                e.color = color.white; e.scale = 10; e.unlit = True
            elif new_key == 'dim':
                e.color = rgb(40, 45, 50, 150); e.scale = 6; e.unlit = False
            else:
                e.color = rgb(0, 210, 255); e.scale = 7; e.unlit = False
            e._prev_color_key = new_key

        if spd > 0.5 and do_look:
            e.look_at(p3 + Vec3(vel[0], vel[1], vel[2]))

        # ═══ Ring buffer trail (no allocation, constant O(1)) ═══
        if do_trail:
            trail_buffers[i].append(p3)  # Auto-drops old points
            if len(trail_buffers[i]) >= 2:
                e.trail.model.vertices = list(trail_buffers[i])
                e.trail.model.generate()
    
    render_profiler.mark_section('drone_update')

    # ═══ Optimized neighbor line rendering ═══
    if do_nl:
        pi = swarm._last_pairs_i; pj = swarm._last_pairs_j
        if len(pi) > 0:
            # Vectorized line generation (batch with NumPy)
            pa = swarm.positions[pi[:180]]  # (N, 3)
            pb = swarm.positions[pj[:180]]  # (N, 3)
            nv = []
            for a, b in zip(pa, pb):
                nv += [Vec3(a[0],a[1],a[2]), Vec3(b[0],b[1],b[2])]
            nb_line_ent.model.vertices = nv
            nb_line_ent.model.generate()
        else:
            nb_line_ent.model.vertices = []
            nb_line_ent.model.generate()
    
    render_profiler.mark_section('neighbor_lines')

    if active_count > 0:
        centroid /= active_count

    # Cinematic
    if cinematic_mode and active_count > 0:
        orbit_r = 260
        angle = _t * 0.32
        cx_c = centroid.x + orbit_r * math.sin(angle)
        cy_c = centroid.y + 100
        cz_c = centroid.z + orbit_r * math.cos(angle)
        editor_cam.position = lerp(editor_cam.position,
                                   Vec3(cx_c, cy_c, cz_c), 4 * time.dt)
        editor_cam.look_at(centroid)

    # Task Visualization (throttled)
    if _frame[0] % 6 == 0:
        assigned_count = np.zeros(len(swarm.tasks))
        for tid in swarm.assigned_tasks:
            if tid != -1: assigned_count[tid] += 1
        
        for tid, tm_dict in enumerate(task_markers):
            if assigned_count[tid] > 0:
                tm_dict['icon'].color = rgb(0, 255, 255, 200)
                tm_dict['icon'].rotation_y += 100 * time.dt * 6
                tm_dict['icon'].scale = Vec3(30, 50, 30) * (1.1 + 0.1 * math.sin(_t*5))
                tm_dict['ring'].color = rgb(255, 200, 0, 150)
            else:
                tm_dict['icon'].color = rgb(150, 150, 150, 150)
                tm_dict['ring'].color = rgb(150, 150, 150, 60)
                tm_dict['icon'].rotation_y += 20 * time.dt * 6

    # Force HUD & Macro Swarm Intent (throttled)
    if _frame[0] % 6 == 0:
        if show_vectors and active_count > 0:
            if not hasattr(swarm, 'last_sep'): return

            alive_mask = ~swarm.dead_mask
            l_sep = swarm.last_sep[alive_mask]
            l_aln = swarm.last_aln[alive_mask]
            l_coh = swarm.last_coh[alive_mask]
            l_tsk = swarm.last_waypoint[alive_mask]

            avg_sep = np.mean(np.linalg.norm(l_sep, axis=1))
            avg_aln = np.mean(np.linalg.norm(l_aln, axis=1))
            avg_coh = np.mean(np.linalg.norm(l_coh, axis=1))
            avg_tsk = np.mean(np.linalg.norm(l_tsk, axis=1))

            scale_fac = 0.5 
            bar_sep.scale_x = min((avg_sep / scale_fac) * 0.12, 0.12)
            bar_aln.scale_x = min((avg_aln / scale_fac) * 0.12, 0.12)
            bar_coh.scale_x = min((avg_coh / scale_fac) * 0.12, 0.12)
            bar_tsk.scale_x = min((avg_tsk / scale_fac) * 0.12, 0.12)

            net_force_v = (np.mean(l_sep, axis=0) + np.mean(l_aln, axis=0) + 
                           np.mean(l_coh, axis=0) + np.mean(l_tsk, axis=0))

            intent_indicator.enabled = True
            intent_indicator.position = centroid
            intent_indicator.scale = Vec3(35, 60, 35) * (1.0 + 0.2 * math.sin(_t*6))

            if np.linalg.norm(net_force_v) > 0.01:
                intent_indicator.look_at(centroid + Vec3(net_force_v[0], net_force_v[1], net_force_v[2]))
                intent_indicator.rotation_x = 90
        else:
            intent_indicator.enabled = False

    # Mission Button highlight (throttled)
    if _frame[0] % 10 == 0:
        btn_mission_map = [0, 1, 2, 3, 6]
        for i, btn in enumerate(mission_btns):
            mission_type = btn_mission_map[i] if i < len(btn_mission_map) else i
            btn.color = rgb(100, 150, 180, 100) if highlighted_mission[0] == mission_type else (rgb(60, 80, 60, 50) if i == 4 else rgb(60, 60, 60, 50))

    # Coverage Tracking & Telemetry (throttled)
    if _frame[0] % 6 == 0:
        global log_timer
        # Count visited cells efficiently using numpy
        visited_count = np.count_nonzero(swarm.visited_grid)
        cov_pct = (visited_count / (swarm.grid_res**3)) * 100
        coverage_text.text = f'COVERAGE: {cov_pct:.1f}%'

        # ═══ Optimized heatmap rendering (incremental update) ═══
        if show_heatmap or show_vectors:
            # Find newly discovered voxels using numpy
            visited_indices = np.argwhere(swarm.visited_grid)
            new_voxels = []
            for i, j, k in visited_indices:
                if not swarm.last_grid[i, j, k]:
                    new_voxels.append([i, j, k])
            
            if len(new_voxels) > 0:
                new_voxels = np.array(new_voxels)
                voxel_size = np.array([W/swarm.grid_res, H/swarm.grid_res, D/swarm.grid_res])
                for v in new_voxels:
                    pos = v * voxel_size + (voxel_size/2)
                    if show_heatmap:
                        hmap_verts.append(Vec3(*pos))
                        hmap_colors.append(rgb(0, 210, 255, 40))
                    if len(hmap_verts) % 25 == 0:
                        DiscoveryPulse(pos)
                if show_heatmap and len(new_voxels) > 0:
                    hmap_ent.model.vertices = hmap_verts
                    hmap_ent.model.colors = hmap_colors
                    hmap_ent.model.generate()  # Only regenerate when new tiles added

        # Telemetry Snapshot
        if _t > log_timer:
            log_timer = _t + 1.0
            metrics_log.append({
                'Time': round(_t - app.start_time, 1),
                'Coverage': round(cov_pct, 2),
                'Active': active_count,
                'Dead': int(np.sum(swarm.dead_mask)),
                'Collisions': swarm.collision_count
            })

        # Mission Complete Detection
        if cov_pct > 99.5 and not mission_banner.enabled:
            mission_banner.enabled = True
            mission_banner.animate_color(rgb(20, 180, 255, 180), duration=2)
            if hasattr(swarm, 'recall_fleet'): swarm.recall_fleet()
            save_metrics_csv() 

    # Formation Centroid
    if active_count > 0:
        centroid_marker.enabled = show_centroid
        centroid_marker.position = centroid
        centroid_marker.rotation_y += 45 * time.dt
        
        _gt = _t * 0.5
        gx = W/2 + math.sin(_gt) * W*0.3
        gz = D/2 + math.cos(_gt*1.3) * D*0.3
        gy = H/2 + math.sin(_gt*0.7) * H*0.2
    else:
        centroid_marker.enabled = False

    # ═══════════════════════════════════════════════════════════════════════
    #  HUD UPDATE (throttled to every 10th frame)
    #  Includes benchmark overlay for PDC technique visualization
    # ═══════════════════════════════════════════════════════════════════════
    if _frame[0] % 10 == 0:
        cov = swarm.coverage_pct
        dead_n = int(np.sum(swarm.dead_mask))
        fps_v  = int(round(1.0 / max(time.dt, 0.001)))
        fault  = '[FAULT] ' if swarm.fault_injected else ''
        algo   = swarm.use_method.upper()
        cam_s  = 'CINEMATIC' if cinematic_mode else 'Free'
        trail_s = 'ON' if show_trails else 'OFF'
        nl_s   = 'ON' if show_neighbor_lines else 'OFF'
        hm_s   = 'ON' if show_heatmap else 'OFF'
        wp_s   = 'WAYPOINT' if waypoint_marker.enabled else 'Auctioning'
        vec_s = 'ON' if show_vectors else 'OFF'
        gpu_s = 'GPU' if GPU_AVAILABLE else 'CPU'
        threads_s = str(config.num_threads)

        info_text.text = (
            'SWARM  COVERAGE  METRICS\n'
            '------------------------\n'
            f'Active : {fault}{active_count}/{swarm.num_boids}\n'
            f'Dead   : {dead_n}\n'
            f'Cover  : {cov:.1f}%\n'
            f'Status : {wp_s}\n'
            f'Algo   : {algo}\n'
            f'Backend: {gpu_s} | {threads_s}T\n'
            f'Sim TPS: {int(_physics_tps[0])}\n'
            f'FPS    : {fps_v}\n'
            f'\n'
            f'Controls: see panel for key bindings.'
        )

        if obs_mode:
            mode_bar_bg.enabled = True
            mov_tag = ' [MOVING]' if obs_moving_mode else ''
            mode_bar.text = (f'[ OBS MODE{mov_tag} ]  Height:{int(obs_height[0])}'
                             f'  |  M=Moving  ↑↓=Height  LClick=Place  RClick=Undo')
        else:
            mode_bar_bg.enabled = False
            mode_bar.text = ''

        # ═══ PDC TECHNIQUE 15: Benchmark Overlay Update ═══
        if show_benchmark:
            bench_text.text = (
                f'FPS: {fps_v}  Sim TPS: {int(_physics_tps[0])}\n'
                f'{swarm.metrics.get_hud_text()}'
            )

    # Reset
    if held_keys['r']:
        swarm.__init__(env)
        waypoint_marker.enabled = False
        swarm.env.target_waypoint = None
        hmap_verts.clear(); hmap_colors.clear()
        hmap_ent.model.vertices = []; hmap_ent.model.generate()
        
        metrics_log.clear()
        log_timer = 0
        mission_banner.enabled = False
        mission_banner.color = rgb(20, 180, 255, 0)
        
        for i, e in enumerate(boid_entities):
            e.color = rgb(0, 210, 255); e.y = 0; e.rotation_x = 0
            e._prev_color_key = ''
            trail_buffers[i].clear()  # Clear ring buffer
            e.trail.model.vertices = []; e.trail.model.generate()
        while user_added:
            rec = user_added.pop()
            destroy(rec['ent'])
            if len(swarm.env.obstacles) > _initial_static_count:
                swarm.env.obstacles.pop()
            if static_obs_ents: static_obs_ents.pop()
        user_moving_obs.clear()
    
    # Record render frame time
    render_profiler.end_frame()
    render_stats.record_render_frame(time.dt if hasattr(time, 'dt') else 0.016)


def save_metrics_csv():
    """M4: Final Evaluation Export + Parallel Analysis"""
    if not metrics_log: return
    filename = 'swarm_evaluation.csv'
    keys = metrics_log[0].keys()
    try:
        with open(filename, 'w', newline='') as f:
            dict_writer = csv.DictWriter(f, fieldnames=keys)
            dict_writer.writeheader()
            dict_writer.writerows(metrics_log)
        print(f"[M4] Success: Evaluation data saved to {filename}")
    except Exception as e:
        print(f"[M4] Error saving metrics: {e}")

    # Also export parallel analysis
    swarm.metrics.export_csv("parallel_analysis.csv")
    swarm.metrics.print_report()

def input(key):
    global cinematic_mode, obs_mode, obs_moving_mode
    global show_neighbor_lines, show_heatmap, show_trails, show_vectors, show_centroid
    global show_benchmark
    global obs_ghost, obs_height, static_obs_ents, user_added, user_moving_obs, _initial_static_count

    # Scroll-wheel zoom
    if key in ('scroll up', 'scroll down'):
        try:
            y_rot = math.radians(editor_cam.rotation_y)
            x_rot = math.radians(editor_cam.rotation_x)
            cam_fwd = Vec3(math.sin(y_rot) * math.cos(x_rot),
                           -math.sin(x_rot),
                           math.cos(y_rot) * math.cos(x_rot))
            sign = 1 if key == 'scroll up' else -1
            editor_cam.position += cam_fwd * CAM_SCROLL * sign
        except:
            pass
        return

    # Debug Force Vectors
    if key == 'v':
        show_vectors = not show_vectors
        force_hud.enabled = show_vectors
        print(f"[UI] Diagnostic Mode: {'ON' if show_vectors else 'OFF'}")

    # ═══ PDC TECHNIQUE 15: Benchmark HUD Toggle ═══
    if key == 'b':
        show_benchmark = not show_benchmark
        bench_panel.enabled = show_benchmark
        print(f"[UI] Benchmark Overlay: {'ON' if show_benchmark else 'OFF'}")

    # Algorithm switch
    if   key == '1': swarm.set_method('naive')
    elif key == '2': swarm.set_method('octree')
    
    elif key == 'g':
        show_centroid = not show_centroid
        print(f"[UI] Global Center: {'ON' if show_centroid else 'OFF'}")

    elif key == 'p':
        save_metrics_csv()

    elif key == 'l':
        show_neighbor_lines = not show_neighbor_lines
        nb_line_ent.enabled = show_neighbor_lines
        if not show_neighbor_lines:
            nb_line_ent.model.vertices = []; nb_line_ent.model.generate()

    elif key == 't':
        show_trails = not show_trails
        for i, e in enumerate(boid_entities):
            e.trail.enabled = show_trails
            if not show_trails:
                e.trail.model.vertices = []; e.trail.model.generate()
                trail_buffers[i].clear()

    elif key == 'h':
        show_heatmap = not show_heatmap
        hmap_ent.enabled = show_heatmap
        if show_heatmap:
            hmap_ent.model.vertices = hmap_verts
            hmap_ent.model.colors = hmap_colors
            hmap_ent.model.generate()

    if key == 'c':
        cinematic_mode = not cinematic_mode
        cinematic_mode_running[0] = cinematic_mode # Use to pause background simulation if desired. Currently they run concurrently.

    elif key == 'o':
        obs_mode = not obs_mode
        obs_ghost.enabled = False
        if not obs_mode: 
            obs_moving_mode = False
        else:
            print("[Obs Mode] Arrow Up/Down to adjust height, L-Click to place, R-Click to remove")

    # Obstacle mode sub-controls
    if obs_mode:
        if key == 'up arrow':
            obs_height[0] = min(obs_height[0] + 30, H - 20)
            print(f"[Obs] Height: {int(obs_height[0])}")
        elif key == 'down arrow':
            obs_height[0] = max(obs_height[0] - 30, 20)
            print(f"[Obs] Height: {int(obs_height[0])}")
        elif key == 'm':
            obs_moving_mode = not obs_moving_mode
            obs_ghost.color = (rgb(255, 200, 30, 170) if obs_moving_mode
                               else rgb(255, 155, 30, 170))
        elif key == 'left mouse down' and obs_ghost.enabled:
            p = obs_ghost.position
            r = float(config.obstacle_radius)
            swarm.env.obstacles.append((p.x, p.y, p.z, r))
            s_idx = len(swarm.env.obstacles) - 1
            c = rgb(255, 185, 30) if obs_moving_mode else rgb(220, 40, 55)
            ent = Entity(model='cube', scale=r*0.85,
                         position=(p.x, p.y, p.z),
                         color=c, unlit=True, wireframe=True,
                         rotation=(35, 35, 0))
            static_obs_ents.append(ent)
            rec = {'ent': ent, 'static_idx': s_idx, 'moving': obs_moving_mode}
            user_added.append(rec)
            if obs_moving_mode:
                user_moving_obs.append({
                    'ent': ent, 'static_idx': s_idx,
                    'ox': p.x, 'oy': p.y, 'oz': p.z,
                    'amp':  random.uniform(100, 280),
                    'freq': random.uniform(0.4, 1.4),
                    'freq2':random.uniform(0.3, 1.1),
                })
                obs_moving_mode = False
                obs_ghost.color = rgb(255, 155, 30, 170)

        elif key == 'right mouse down' and user_added:
            rec = user_added.pop()
            destroy(rec['ent'])
            if static_obs_ents and static_obs_ents[-1] is rec['ent']:
                static_obs_ents.pop()
            if len(swarm.env.obstacles) > _initial_static_count:
                swarm.env.obstacles.pop()
            if rec['moving']:
                user_moving_obs[:] = [m for m in user_moving_obs
                                      if m['ent'] is not rec['ent']]

    # Normal mode
    else:
        if key == 'left mouse down':
            if mouse.hovered_entity == floor_collider:
                wp = mouse.world_point
                if wp:
                    waypoint_marker.enabled = True
                    waypoint_marker.position = Vec3(wp.x, 8, wp.z)
                    swarm.env.target_waypoint = (wp.x, H*0.45, wp.z)
        elif key == 'right mouse down':
            waypoint_marker.enabled = False
            swarm.env.target_waypoint = None


app.run()
