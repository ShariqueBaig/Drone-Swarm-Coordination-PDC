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
  P              Toggle Fleet Command task panel
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
app.setBackgroundColor(rgb(4, 6, 13))

# Discovery Pulse Effect (M2.7 - High Performance)
# Disabled to avoid "spawning out of nowhere" effect

window.fps_counter.enabled = False
window.entity_counter.enabled = False
window.exit_button.visible = False
window.collider_counter.enabled = False
window.cog_menu.enabled = False
window.vsync = False
application.target_fps = 0

env   = Environment3D()
swarm = SwarmManager3D(env)
W, H  = config.width, config.height
D     = W
D     = env.depth

# ── CAMERA ────────────────────────────────────────────────────────────────────
editor_cam = EditorCamera()
# Position camera to view the complete simulation cube from a natural middle perspective
editor_cam.position = Vec3(W/2, H * 1.5, -D * 2.2)
editor_cam.rotation = (25, 0, 0)
editor_cam.target = Vec3(W/2, H/2, D/2)
editor_cam.pan_speed = Vec2(-5, -5) # Fix inverted pan
camera.clip_plane_far = 20000
cinematic_mode = False
show_mission_hud = True
_saved_cam_z = [0]
CAM_PAN  = 800
CAM_ZOOM = 1200
CAM_SCROLL = 400
cam_velocity = Vec3(0, 0, 0)

# ── LIGHTING ──────────────────────────────────────────────────────────────────
d_light = DirectionalLight(parent=scene, y=8, z=4, color=rgb(160, 185, 255))
d_light.look_at(Vec3(0, -1, 0.4))
AmbientLight(parent=scene, color=rgb(18, 22, 38))

# ── SKYBOX ────────────────────────────────────────────────────────────────────
sky_sphere = Entity(model='sphere', scale=15000, color=rgb(4, 6, 13),
       double_sided=True, unlit=True)
random.seed(77)
_sv = [Vec3(random.uniform(-6000,6000), random.uniform(80,4500),
            random.uniform(-6000,6000)) for _ in range(100)]
star_field = Entity(model=Mesh(vertices=_sv, mode='point', thickness=3),
       color=rgb(195, 215, 255), unlit=True)

# ── FLOOR ─────────────────────────────────────────────────────────────────────
floor_plane = Entity(model='plane', scale=(W, 1, W), position=(W/2, 0, W/2),
       color=rgb(15, 28, 48), unlit=True)



floor_collider = Entity(model='plane', scale=(30000, 1, 30000),
                        position=(W/2, 0, W/2),
                        collider='box', visible=False)

# ── BOUNDARY ─────────────────────────────────────────────────────────────────
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

# ── FORMATION CENTROID & GLOBAL CENTER ────────────────────────────────────────
centroid_marker = Entity(model='diamond', scale=(30, 50, 30), color=rgb(255, 255, 255, 180), 
                         unlit=True, wireframe=True, enabled=False)

# ── NEIGHBOR LINES ────────────────────────────────────────────────────────────
show_neighbor_lines = False
nb_line_ent = Entity(model=Mesh(vertices=[], mode='line', thickness=1),
                     color=rgb(0, 190, 255, 70), unlit=True, enabled=False)

# ── TRAILS ────────────────────────────────────────────────────────────────────
show_trails = False
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
boid_entities = []
batch_renderer = BatchDroneRenderer(swarm.num_boids, max_trail_len=10)
trail_buffers = [deque(maxlen=10) for _ in range(swarm.num_boids)]

for i in range(swarm.num_boids):
    drone = Entity(model='sphere', scale=7, color=rgb(0, 210, 255), unlit=True)
    drone.trail = Entity(model=Mesh(vertices=[], mode='line', thickness=2),
                         color=rgb(0, 145, 220, 185), unlit=True)
    drone._prev_color_key = ''
    boid_entities.append(drone)

optimized_heatmap = OptimizedHeatmapRenderer(swarm.grid_res, W)
optimized_neighbors = OptimizedNeighborLineRenderer(max_pairs=360)
render_profiler = RenderProfiler()
render_stats = RenderFrameStats()
    
# ── M3 TASK MARKERS & ALLOCATOR PRISMS ───────────────────────────────────────
task_markers = []
# Prisms disabled as heatmaps suffice for area coverage


# ── CARGO OBJECT & MARKERS (M3 Task) ─────────────────────────────────────────
cargo_box = Entity(model='cube', scale=40, color=rgb(255, 165, 30, 180),
                   unlit=True, wireframe=True, enabled=False)

cargo_core = Entity(parent=cargo_box, model='diamond', scale=0.6, 
                    color=rgb(255, 200, 100, 220), unlit=True)
cargo_ring1 = Entity(parent=cargo_box, model='circle', scale=1.3, rotation_x=45,
                     color=rgb(255, 165, 30, 150), unlit=True)
cargo_ring2 = Entity(parent=cargo_box, model='circle', scale=1.3, rotation_z=45, rotation_y=90,
                     color=rgb(255, 165, 30, 150), unlit=True)

dropoff_pad = Entity(model='cylinder', scale=(70, 20, 70), color=rgb(30, 255, 120, 80),
                     unlit=True, enabled=False)
dropoff_ring = Entity(parent=dropoff_pad, model='circle', scale=1.1, rotation_x=90, 
                      y=0.55, color=rgb(30, 255, 120, 200), unlit=True)
dropoff_beam = Entity(parent=dropoff_pad, model='cylinder', scale=(0.8, 10, 0.8), y=5, 
                      color=rgb(30, 255, 120, 30), unlit=True)

# ── HUD ───────────────────────────────────────────────────────────────────────
ui_panel = Entity(parent=camera.ui, model='quad', scale=(0.3, 0.94),
                  position=(-0.75, 0.0), color=rgb(8, 12, 22, 180))
ui_panel_bg = Entity(parent=camera.ui, model='quad', scale=(0.305, 0.945),
       position=(-0.75, 0.0), color=rgb(0, 140, 255, 15), z=1)

info_text = Text(text='Initializing...', position=(-0.88, 0.44),
                 scale=0.75, color=color.white, background=False)

if not hasattr(config, 'waypoint_weight'):
    config.waypoint_weight = 2.5

if not hasattr(config, 'transport_drone_count'):
    config.transport_drone_count = 10

slider_x = -0.85
slider_start_y = 0.08
def _make_slider(text, val, y_off):
    heading_x = slider_x + 0.02
    txt = Text(parent=camera.ui, text=text, position=(heading_x, y_off + 0.035),
        scale=0.65, color=color.white, origin=(-0.5,0))
    s = ThinSlider(text='', dynamic=True, min=0, max=10, default=val,
                x=slider_x, y=y_off - 0.02, parent=camera.ui, scale=0.32)
    s.text_entity = txt
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
    text='<cyan>KEY COMMANDS\n'
         '<gray>1/2  : Algo Switch\n'
         'L/T/H: Debug Toggles\n'
         'C/P/B: View Toggles\n'
         'O    : Obstacle Mode\n'
         'R    : Reset Swarm\n'
         'V/G  : Diagnostic View\n'
         'WASD : Camera Pan\n'
         'QE   : Elevation\n'
         'Scroll: Move Forward',
    position=(-0.86, -0.22), scale=0.6, color=color.azure
)

show_vectors = False
intent_indicator = Entity(model='diamond', scale=(35, 60, 35), color=rgb(0, 210, 255, 120),
                          unlit=True, wireframe=True, enabled=False)
intent_glow = Entity(parent=intent_indicator, model='circle', scale=1.2, rotation_x=90,
                     color=rgb(0, 210, 255, 60), unlit=True)

# ── FORCE BREAKDOWN HUD ──────────────────────────────────────────────────────
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

mode_bar_bg = Entity(parent=camera.ui, model='quad', scale=(0.6, 0.06), position=(0, 0.45), color=rgb(8, 12, 22, 140), enabled=False)
mode_bar = Text(text='', origin=(0,0), position=(0, 0.45),
                scale=1.0, color=rgb(255, 165, 30), background=False)

boundary_faces = Entity(model='cube', scale=(-W, H, D), position=(W/2, H/2, D/2),
                        color=rgb(15, 28, 48, 160), unlit=True, enabled=True)

scan_plane = Entity(model='quad', scale=(W, D), rotation_y=90, 
              color=rgb(0, 255, 255, 10), unlit=True, enabled=False)

# ── MISSION FLEET HUD (FIXED SPACING) ────────────────────────────────────────
mission_hud_panel = Entity(parent=camera.ui, model='quad', scale=(0.22, 0.88),
                           position=(0.72, 0.0), color=rgb(8, 12, 22, 200))
mission_hud_border = Entity(parent=camera.ui, model='quad', scale=(0.225, 0.885),
       position=(0.72, 0.0), color=rgb(0, 140, 255, 40), z=0.01)

# Panel Content (Centered Mission Selection)
# No Header Text (Requested Removal)


# Fault & Reset (Centered Context)
fault_btn = Button(parent=mission_hud_panel, text='INJECT FAULT',
                   scale=(0.82, 0.06), position=(0, -0.26), color=rgb(80, 45, 45, 160))
reset_btn = Button(parent=mission_hud_panel, text='RESET FLEET',
                   scale=(0.82, 0.06), position=(0, -0.34), color=rgb(45, 70, 50, 160))

def inject_fault():
    swarm.inject_faults(0.2)
    print("[M3] Chaos Injected: 20% Drones failing!")
fault_btn.on_click = inject_fault

def perform_full_reset():
    global log_timer, cinematic_mode
    swarm.__init__(env)
    waypoint_marker.enabled = False
    swarm.env.target_waypoint = None
    hmap_verts.clear(); hmap_colors.clear()
    hmap_ent.model.vertices = []; hmap_ent.model.generate()
    
    metrics_log.clear()
    log_timer = 0
    mission_banner.enabled = False
    mission_banner.has_shown = False
    banner_text.color = color.rgba(255,255,255,0)
    banner_subtext.color = color.rgba(0, 255, 255, 0)
    highlighted_mission[0] = -1
    cinematic_mode = False
    
    if hasattr(cargo_box, 'initialized'):
        del cargo_box.initialized

    for i, e in enumerate(boid_entities):
        e.color = rgb(0, 210, 255); e.y = 0; e.rotation_x = 0
        e._prev_color_key = ''
        trail_buffers[i].clear()
        e.trail.model.vertices = []; e.trail.model.generate()
        
    while user_added:
        rec = user_added.pop()
        destroy(rec['ent'])
        if len(swarm.env.obstacles) > _initial_static_count:
            swarm.env.obstacles.pop()
        if static_obs_ents: static_obs_ents.pop()
    user_moving_obs.clear()
    print("[M3] Simulation Reset: Fleet and Tasks Ready.")

def reset_fleet():
    perform_full_reset()

reset_btn.on_click = reset_fleet

highlighted_mission = [-1] 

def select_mission(m_id):
    if highlighted_mission[0] == m_id: 
        highlighted_mission[0] = -1
    else: 
        highlighted_mission[0] = m_id
        alive = ~swarm.dead_mask
        
        if m_id == 7:
            alive_indices = np.where(alive)[0]
            num_to_assign = min(4, len(alive_indices))
            
            if num_to_assign > 0:
                pickup_point = swarm.tasks[8]
                distances = np.linalg.norm(
                    swarm.positions[alive_indices] - pickup_point, 
                    axis=1
                )
                closest_indices = np.argsort(distances)[:num_to_assign]
                selected_drones = alive_indices[closest_indices]
                
                assign_mask = np.zeros(swarm.num_boids, dtype=bool)
                assign_mask[selected_drones] = True
                
                swarm.mission_type[assign_mask] = m_id
                swarm.assigned_tasks[assign_mask] = -1
                swarm.transport_phase[assign_mask] = 0
                swarm.delivered_mask[assign_mask] = False
                
                remaining = alive_indices[~np.isin(alive_indices, selected_drones)]
                if len(remaining) > 0:
                    swarm.mission_type[remaining] = 3
                    swarm.assigned_tasks[remaining] = -1
                
                print(f"[M3] Transport Mission: {num_to_assign} drones assigned")
            else:
                print("[M3] No drones available for transport")
        else:
            swarm.mission_type[alive] = m_id
            swarm.assigned_tasks[alive] = -1
            
        print(f"[M3] Fleet Mission Updated: {m_id}")

# ── MISSION BUTTONS (Centered Y) ─────────────────────────────────────────────
mission_btns = []
mission_labels = ['Idle / Flocking', 'Object Transport', 'Area Coverage', 'Recall Fleet']
btn_mission_map = [3, 7, 6, 5]

for i, label in enumerate(mission_labels):
    btn = Button(parent=mission_hud_panel, text=label, scale=(0.85, 0.07),
                 position=(0, 0.12 - i * 0.09), color=rgb(60, 60, 60, 180),
                 on_click=Func(select_mission, btn_mission_map[i]))
    mission_btns.append(btn)

# Mission Success Banner
# ── MISSION SUCCESS BANNER (Premium Redesign) ─────────────────────────────────
mission_banner = Entity(parent=camera.ui, enabled=False, z=-10)
mission_banner.has_shown = False

banner_bg = Entity(parent=mission_banner, model='quad', scale=(0.8, 0.15), 
                   position=(0, 0.35), color=rgb(8, 12, 22, 0))
banner_line_t = Entity(parent=mission_banner, model='quad', scale=(0, 0.004), 
                       position=(0, 0.425), color=rgb(0, 255, 255, 0))
banner_line_b = Entity(parent=mission_banner, model='quad', scale=(0, 0.004), 
                       position=(0, 0.275), color=rgb(0, 255, 255, 0))
banner_text = Text(parent=mission_banner, text='MISSION ACCOMPLISHED', 
                   origin=(0,0), position=(0, 0.36), scale=3, color=color.rgba(255,255,255,0))
banner_subtext = Text(parent=mission_banner, text='AREA 100% COVERED • RECALL PROTOCOL ACTIVE', 
                      origin=(0,0), position=(0, 0.31), scale=1.2, color=color.rgba(0, 255, 255, 0))

# ── CINEMATIC HUD (M3 Extension) ─────────────────────────────────────────────
cinematic_hud = Entity(parent=camera.ui, enabled=False)
cine_bar = Entity(parent=cinematic_hud, model='quad', scale=(1.0, 0.05), 
                  position=(0, -0.465), color=rgb(8, 12, 22, 180))
cine_line = Entity(parent=cinematic_hud, model='quad', scale=(1.0, 0.003),
                   position=(0, -0.44), color=rgb(0, 210, 255, 120))
cine_stats = Text(parent=cinematic_hud, text='', position=(0, -0.468), 
                  scale=0.85, color=color.white, origin=(0, 0))

def hide_banner():
    if not mission_banner.enabled: return
    banner_bg.animate_color(rgb(8, 12, 22, 0), duration=1)
    banner_line_t.animate_scale((0, 0.004), duration=1)
    banner_line_b.animate_scale((0, 0.004), duration=1)
    banner_text.animate_color(color.rgba(255,255,255,0), duration=1)
    banner_subtext.animate_color(color.rgba(0, 255, 255, 0), duration=1)
    invoke(setattr, mission_banner, 'enabled', False, delay=1.1)

def show_banner():
    if getattr(mission_banner, 'has_shown', False): return
    global cinematic_mode
    mission_banner.enabled = True
    mission_banner.has_shown = True
    cinematic_mode = True 
    highlighted_mission[0] = 5 
    
    banner_bg.animate_color(rgb(8, 12, 22, 220), duration=0.6)
    banner_line_t.animate_scale((0.8, 0.004), duration=0.8, curve=curve.out_expo)
    banner_line_b.animate_scale((0.8, 0.004), duration=0.8, curve=curve.out_expo)
    banner_line_t.animate_color(rgb(0, 255, 255, 255), duration=0.6)
    banner_line_b.animate_color(rgb(0, 255, 255, 255), duration=0.6)
    banner_text.animate_color(color.white, duration=0.8)
    banner_subtext.animate_color(rgb(0, 255, 255), duration=0.8)
    
    invoke(hide_banner, delay=5)

# ═══════════════════════════════════════════════════════════════════════════════
show_benchmark = False
bench_panel = Entity(parent=camera.ui, enabled=False)

Entity(parent=bench_panel, model='quad', scale=(0.42, 0.26),
       position=(0.0, -0.38), color=rgb(8, 12, 22, 190), z=0.02)
Entity(parent=bench_panel, model='quad', scale=(0.425, 0.265),
       position=(0.0, -0.38), color=rgb(0, 255, 180, 15), z=0.01)

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
import threading

cinematic_mode_running = [False]
_physics_tps = [0.0]

def _physics_worker():
    last_t = time.perf_counter()
    frames = 0
    while True:
        try:
            swarm.update()
            frames += 1
            curr_t = time.perf_counter()
            if curr_t - last_t >= 1.0:
                _physics_tps[0] = frames / (curr_t - last_t)
                frames = 0
                last_t = curr_t
        except RuntimeError as e:
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
#  UPDATE LOOP
# ═══════════════════════════════════════════════════════════════════════════════
def update():
    _frame[0] += 1
    _t = time.time()
    
    try:
        cov_pct = swarm.coverage_pct
    except Exception:
        cov_pct = 0.0

    try:
        y_rot = math.radians(editor_cam.rotation_y)
        x_rot = math.radians(editor_cam.rotation_x)
        fwd   = Vec3(math.sin(y_rot) * math.cos(x_rot),
                     -math.sin(x_rot),
                     math.cos(y_rot) * math.cos(x_rot))
        right = Vec3(math.cos(y_rot), 0, -math.sin(y_rot))
    except:
        fwd, right = Vec3(0,0,1), Vec3(1,0,0)

    global cam_velocity
    desired_vel = Vec3(0, 0, 0)

    if not cinematic_mode:
        if held_keys['a']:   desired_vel -= right * CAM_PAN
        if held_keys['d']:   desired_vel += right * CAM_PAN
        if held_keys['w']:   desired_vel += fwd * CAM_PAN
        if held_keys['s']:   desired_vel -= fwd * CAM_PAN
        if held_keys['e']:   desired_vel += Vec3(0, CAM_PAN, 0)
        if held_keys['q']:   desired_vel -= Vec3(0, CAM_PAN, 0)
        if held_keys['=']:   desired_vel += fwd * CAM_ZOOM
        if held_keys['-']:   desired_vel -= fwd * CAM_ZOOM

        # Smooth, natural acceleration and deceleration
        cam_velocity = lerp(cam_velocity, desired_vel, time.dt * 10)
        editor_cam.position += cam_velocity * time.dt
    else:
        cam_velocity = Vec3(0, 0, 0)

    if not hasattr(update, '_transport_last_press'):
        update._transport_last_press = 0
    
    now = time.time()
    if held_keys['8'] and (now - update._transport_last_press) > 0.3:
        config.transport_drone_count = max(1, config.transport_drone_count - 5)
        update._transport_last_press = now
        
    if held_keys['9'] and (now - update._transport_last_press) > 0.3:
        config.transport_drone_count = min(swarm.num_boids, config.transport_drone_count + 5)
        update._transport_last_press = now
    
    if waypoint_marker.enabled:
        waypoint_marker.rotation_y += 70 * time.dt
        if _frame[0] % 4 == 0:
            waypoint_ring.scale_x = 3
            waypoint_ring.scale_y = 3

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

    for mo in user_moving_obs:
        nx = mo['ox'] + mo['amp'] * math.sin(mo['freq']  * _t)
        nz = mo['oz'] + mo['amp'] * math.cos(mo['freq2'] * _t)
        nx = max(50, min(W-50, nx))
        nz = max(50, min(W-50, nz))
        sidx = mo['static_idx']
        if sidx < len(swarm.env.obstacles):
            old = swarm.env.obstacles[sidx]
            swarm.env.obstacles[sidx] = (nx, old[1], nz, old[3])

    for i, ob in enumerate(swarm.env.obstacles):
        if i < len(static_obs_ents):
            static_obs_ents[i].position = (ob[0], ob[1], ob[2])
    for i, d in enumerate(swarm.env.dynamic_obstacles):
        if i < len(dyn_obs_ents):
            dyn_obs_ents[i].position = (d.x, d.y, d.z)

    render_profiler.start_frame()
    active_count = 0
    centroid = Vec3(0, 0, 0)
    do_hmap  = show_heatmap and (_frame[0] % 6 == 0)
    do_trail = show_trails  and (_frame[0] % 8 == 0)
    do_nl    = show_neighbor_lines and (_frame[0] % 10 == 0)
    do_look  = (_frame[0] % 2 == 0)

    with swarm.state_lock:
        local_positions = swarm.positions.copy()
        local_velocities = swarm.velocities.copy()
        local_dead = swarm.dead_mask.copy()
        local_failed = swarm.failed_mask.copy()
        local_missions = swarm.mission_type.copy()
        local_delivered = swarm.delivered_mask.copy()
    
    render_profiler.mark_section('state_copy')
    
    for i in range(len(boid_entities)):
        e = boid_entities[i]
        pos = local_positions[i]
        is_dead = local_dead[i]
        is_failed = local_failed[i]

        if is_dead or is_failed:
            if e._prev_color_key != 'dead':
                e.color = rgb(255, 30, 30)
                e._prev_color_key = 'dead'
            e.y = max(-15, e.y - 140*time.dt)
            e.rotation_x += 75*time.dt
            continue

        active_count += 1
        p3 = Vec3(pos[0], pos[1], pos[2])
        e.position = p3
        centroid += p3

        vel = local_velocities[i]
        spd = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        h_id = highlighted_mission[0]
        
        if local_missions[i] == 7:
            phase = swarm.transport_phase[i]
            if phase == 0:
                new_key = 'transport_goto'
            elif phase == 1:
                new_key = 'transport_carry'
            else:
                new_key = 'delivered'
        elif show_vectors:
            new_key = 'vec'
        elif h_id != -1 and local_missions[i] == h_id:
            new_key = 'sel'
        elif h_id == 7:
            new_key = 'norm'
        elif h_id != -1:
            new_key = 'dim'
        else:
            new_key = 'norm'
        
        if e._prev_color_key != new_key:
            if new_key == 'delivered':
                e.color = rgb(30, 255, 120); e.scale = 9; e.unlit = True
            elif new_key == 'transport_goto':
                e.color = rgb(255, 165, 30); e.scale = 9; e.unlit = True
            elif new_key == 'transport_carry':
                e.color = rgb(255, 50, 50); e.scale = 10; e.unlit = True
            elif new_key == 'vec':
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

        if do_trail:
            trail_buffers[i].append(p3)
            if len(trail_buffers[i]) >= 2:
                e.trail.model.vertices = list(trail_buffers[i])
                e.trail.model.generate()
    
    render_profiler.mark_section('drone_update')

    if do_nl:
        pi = swarm._last_pairs_i; pj = swarm._last_pairs_j
        if len(pi) > 0:
            pa = swarm.positions[pi[:180]]
            pb = swarm.positions[pj[:180]]
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

    if cinematic_mode and active_count > 0:
        cube_center = Vec3(W/2, H/2, D/2)
        dt = max(time.dt, 0.0001)  # Guard against zero dt
        
        # ── First frame: detach camera and neutralize EditorCamera ──
        if camera.parent == editor_cam:
            _saved_cam_z[0] = camera.z
            # Snapshot world transform before reparenting
            wp = Vec3(camera.world_position)
            camera.world_parent = scene
            camera.position = wp
            
            # Neutralize EditorCamera by replacing its update/input with no-ops
            # (cannot use .enabled=False as that also disables child camera entity)
            if not hasattr(editor_cam, '_orig_update'):
                editor_cam._orig_update = editor_cam.update
                editor_cam._orig_input = editor_cam.input
            editor_cam.update = lambda: None
            editor_cam.input = lambda key: None
            
            # Initialize orbit state from current position
            offset = wp - cube_center
            camera._cine_angle = math.atan2(offset.x, offset.z)
            camera._cine_radius = max(math.sqrt(offset.x**2 + offset.z**2), 100)
            camera._cine_y = wp.y
            camera._cine_fov = camera.fov
            camera._cine_time = 0.0

        camera._cine_time += dt
        
        # ── Orbit parameters converge toward targets ──
        target_radius = math.sqrt(W**2 + H**2 + D**2) * 1.05
        target_y = cube_center.y + 600
        
        # Exponential convergence (frame-rate independent)
        camera._cine_radius += (target_radius - camera._cine_radius) * (1.0 - math.exp(-1.2 * dt))
        camera._cine_y      += (target_y - camera._cine_y)           * (1.0 - math.exp(-1.0 * dt))
        
        # Steady orbit rotation
        camera._cine_angle += 0.12 * dt
        
        # ── Set position directly — no intermediate blending ──
        cam_x = cube_center.x + camera._cine_radius * math.sin(camera._cine_angle)
        cam_z = cube_center.z + camera._cine_radius * math.cos(camera._cine_angle)
        cam_y = camera._cine_y
        camera.position = Vec3(cam_x, cam_y, cam_z)
        
        # ── Set rotation directly — compute exact yaw/pitch to look at cube_center ──
        dx = cube_center.x - cam_x
        dy = cube_center.y - cam_y
        dz = cube_center.z - cam_z
        dist_xz = math.sqrt(dx*dx + dz*dz)
        
        camera.rotation_y = math.degrees(math.atan2(dx, dz))
        camera.rotation_x = math.degrees(math.atan2(-dy, dist_xz)) if dist_xz > 0.01 else 0
        camera.rotation_z = 0
        
        # ── FOV converge ──
        camera._cine_fov += (65 - camera._cine_fov) * (1.0 - math.exp(-2.0 * dt))
        camera.fov = camera._cine_fov
    else:
        # Restore EditorCamera methods and re-attach
        if camera.parent != editor_cam:
            if hasattr(editor_cam, '_orig_update'):
                editor_cam.update = editor_cam._orig_update
                editor_cam.input = editor_cam._orig_input
            camera.parent = editor_cam
            camera.position = (0, 0, _saved_cam_z[0])
            camera.rotation = (0, 0, 0)
            
        # Gracefully return FOV to standard when exiting cinematic mode
        if camera.fov > 40.5:
            camera.fov = lerp(camera.fov, 40, 2.5 * time.dt)

    if _frame[0] % 6 == 0:
        assigned_count = np.zeros(len(swarm.tasks))
        for tid in swarm.assigned_tasks:
            if tid != -1: assigned_count[tid] += 1
        
        for tid, tm_dict in enumerate(task_markers):
            if tid >= len(assigned_count): continue
            
            if assigned_count[tid] > 0:
                tm_dict['base'].enabled = True
                tm_dict['icon'].color = rgb(0, 255, 255, 200)
                tm_dict['icon'].rotation_y += 100 * time.dt * 6
                tm_dict['icon'].scale = Vec3(30, 50, 30) # No pulsing for stable visuals
                tm_dict['ring'].color = rgb(255, 200, 0, 150)
            else:
                tm_dict['base'].enabled = False
                tm_dict['icon'].color = rgb(150, 150, 150, 150)
                tm_dict['ring'].color = rgb(150, 150, 150, 60)
                tm_dict['icon'].rotation_y += 20 * time.dt * 6

    transporting = (swarm.mission_type == 7) & (swarm.transport_phase == 1)
    delivered = (swarm.mission_type == 7) & (swarm.transport_phase == 2)
    preparing = (swarm.mission_type == 7) & (swarm.transport_phase == 0)
    
    if np.any(delivered):
        cargo_box.enabled = True
        dropoff_pad.enabled = True
        dropoff = swarm.tasks[9]
        dropoff_pad.position = Vec3(dropoff[0], dropoff[1] - 40, dropoff[2])
        
        timer_vals = swarm.mission_timer[delivered]
        ratio = max(0, timer_vals[0] / 5.0) if len(timer_vals) > 0 else 0
        
        c_pos = np.mean(swarm.positions[delivered], axis=0)
        target_vec = Vec3(c_pos[0], c_pos[1], c_pos[2])
        if not hasattr(cargo_box, 'initialized'):
            cargo_box.position = target_vec
            cargo_box.initialized = True
        else:
            cargo_box.position = lerp(cargo_box.position, target_vec, 12 * time.dt)
            
        # Sink and fade out
        cargo_box.scale_y = max(0.1, 40 * ratio)
        cargo_box.y -= (40 - cargo_box.scale_y) / 2
        
        alpha_outer = int(180 * ratio)
        alpha_core = int(220 * ratio)
        alpha_rings = int(150 * ratio)
        
        cargo_box.color = rgb(30, 255, 120, alpha_outer)
        cargo_core.color = rgb(100, 255, 180, alpha_core)
        cargo_ring1.color = rgb(30, 255, 120, alpha_rings)
        cargo_ring2.color = rgb(30, 255, 120, alpha_rings)
        
        cargo_box.rotation_y += 45 * time.dt
        cargo_core.rotation_y -= 90 * time.dt
        cargo_ring1.rotation_z += 60 * time.dt
        cargo_ring2.rotation_x += 60 * time.dt
    elif np.any(transporting):
        cargo_box.enabled = True
        dropoff_pad.enabled = True
        dropoff = swarm.tasks[9]
        dropoff_pad.position = Vec3(dropoff[0], dropoff[1] - 40, dropoff[2])
        
        cargo_box.scale_y = 40
        c_pos = np.mean(swarm.positions[transporting], axis=0)
        target_vec = Vec3(c_pos[0], c_pos[1], c_pos[2])
        if not hasattr(cargo_box, 'initialized'):
            cargo_box.position = target_vec
            cargo_box.initialized = True
        else:
            cargo_box.position = lerp(cargo_box.position, target_vec, 12 * time.dt)
            
        cargo_box.color = rgb(255, 50, 50, 180)
        cargo_core.color = rgb(255, 100, 100, 220)
        cargo_ring1.color = rgb(255, 50, 50, 150)
        cargo_ring2.color = rgb(255, 50, 50, 150)
        
        cargo_box.rotation_y += 45 * time.dt
        cargo_core.rotation_y -= 90 * time.dt
        cargo_ring1.rotation_z += 60 * time.dt
        cargo_ring2.rotation_x += 60 * time.dt
    elif np.any(preparing):
        cargo_box.enabled = True
        dropoff_pad.enabled = False
        cargo_box.scale_y = 40
        pickup = swarm.tasks[8]
        cargo_box.position = Vec3(pickup[0], pickup[1], pickup[2])
        cargo_box.initialized = True
        
        cargo_box.color = rgb(255, 165, 30, 180)
        cargo_core.color = rgb(255, 200, 100, 220)
        cargo_ring1.color = rgb(255, 165, 30, 150)
        cargo_ring2.color = rgb(255, 165, 30, 150)
        
        cargo_box.rotation_y += 45 * time.dt
        cargo_core.rotation_y -= 90 * time.dt
        cargo_ring1.rotation_z += 60 * time.dt
        cargo_ring2.rotation_x += 60 * time.dt
    else:
        cargo_box.enabled = False
        dropoff_pad.enabled = False

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

    if _frame[0] % 10 == 0:
        btn_mission_map = [3, 7, 6, 5]
        for i, btn in enumerate(mission_btns):
            if i < len(btn_mission_map):
                mission_type = btn_mission_map[i]
                btn.color = rgb(100, 150, 180, 100) if highlighted_mission[0] == mission_type else rgb(60, 60, 60, 50)

    if _frame[0] % 6 == 0:
        global log_timer
        cov_pct = swarm.coverage_pct

        if show_heatmap or show_vectors:
            new_mask = swarm.visited_grid & ~swarm.last_grid
            new_indices = np.argwhere(new_mask)
            
            if len(new_indices) > 0:
                voxel_size = np.array([W/swarm.grid_res, H/swarm.grid_res, D/swarm.grid_res])
                pos_batch = new_indices * voxel_size + (voxel_size/2)
                
                for pos in pos_batch:
                    if show_heatmap:
                        hmap_verts.append(Vec3(*pos))
                        hmap_colors.append(rgb(0, 210, 255, 40))

                if show_heatmap:
                    min_len = min(len(hmap_verts), len(hmap_colors))
                    if min_len > 0:
                        hmap_ent.model.vertices = hmap_verts[:min_len]
                        hmap_ent.model.colors = hmap_colors[:min_len]
                        try:
                            hmap_ent.model.generate()
                        except:
                            pass

        if _t > log_timer:
            log_timer = _t + 1.0
            rob = swarm.get_robustness_score(_t - app.start_time)
            swarm.metrics.record_robustness(rob)
            metrics_log.append({
                'Time': round(_t - app.start_time, 1),
                'Coverage': round(cov_pct, 2),
                'Active': active_count,
                'Dead': int(np.sum(swarm.dead_mask)),
                'Collisions': swarm.collision_count,
                'Robustness': swarm.get_robustness_score(_t - app.start_time)
            })

        if cov_pct > 99.5 and not getattr(mission_banner, 'has_shown', False):
            show_banner()
            if hasattr(swarm, 'recall_fleet'): swarm.recall_fleet()
            save_metrics_csv() 

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

    if _frame[0] % 10 == 0:
        cov = swarm.coverage_pct
        dead_n = int(np.sum(swarm.dead_mask | swarm.failed_mask))
        fps_v  = int(round(1.0 / max(time.dt, 0.001)))
        fault  = '[FAULT] ' if np.any(swarm.failed_mask) else ''
        algo   = swarm.use_method.upper()
        cam_s  = 'CINEMATIC' if cinematic_mode else 'Free'
        trail_s = 'ON' if show_trails else 'OFF'
        nl_s   = 'ON' if show_neighbor_lines else 'OFF'
        hm_s   = 'ON' if show_heatmap else 'OFF'
        wp_s   = 'WAYPOINT' if waypoint_marker.enabled else 'Auctioning'
        vec_s = 'ON' if show_vectors else 'OFF'
        gpu_s = 'GPU' if GPU_AVAILABLE else 'CPU'
        threads_s = str(config.num_threads)

        transport_active = int(np.sum((swarm.mission_type == 7) & ~swarm.dead_mask & ~swarm.failed_mask))
        info_text.text = (
            f'<cyan>SWARM SYSTEM\n'
            f'-------------------\n'
            f'<gray>Active  : <white>{fault}{active_count}/{swarm.num_boids}\n'
            f'<gray>Offline : <white>{dead_n}\n'
            f'<gray>Tasking : <white>{transport_active}/{config.transport_drone_count}\n\n'
            f'<cyan>PERFORMANCE\n'
            f'-------------------\n'
            f'<gray>Algo    : <white>{algo}\n'
            f'<gray>Compute : <white>{gpu_s} | {threads_s}T\n\n'
            f'<cyan>INTEGRITY\n'
            f'-------------------\n'
            f'<gray>Stability: <white>{swarm.get_robustness_score(_t - app.start_time):.2f}'
        )

        if cinematic_mode:
            ui_panel.enabled = False
            ui_panel_bg.enabled = False
            mission_hud_panel.enabled = show_mission_hud
            mission_hud_border.enabled = show_mission_hud
            info_text.enabled = False
            controls_text.enabled = False
            force_hud.enabled = False
            mode_bar_bg.enabled = False
            separation_slider.enabled = False
            alignment_slider.enabled = False
            cohesion_slider.enabled = False
            waypoint_slider.enabled = False
            separation_slider.text_entity.enabled = False
            alignment_slider.text_entity.enabled = False
            cohesion_slider.text_entity.enabled = False
            waypoint_slider.text_entity.enabled = False
            boundary_faces.enabled = True
            boundary_faces.color = rgb(15, 28, 48, 40) # Very slight surface coloring in cinematic mode
            sky_sphere.enabled = False
            star_field.enabled = False
            floor_plane.enabled = False

            # Cinematic HUD Update
            cinematic_hud.enabled = True
            cine_stats.text = (
                f'<cyan>ACTIVE: <white>{active_count}    '
                f'<cyan>OFFLINE: <white>{dead_n}    '
                f'<cyan>FPS: <white>{fps_v}    '
                f'<cyan>TPS: <white>{int(_physics_tps[0])}    '
                f'<cyan>ALGO: <white>{algo}    '
                f'<cyan>COVERAGE: <white>{cov:.1f}%'
            )
        else:
            cinematic_hud.enabled = False
            ui_panel.enabled = True
            ui_panel_bg.enabled = True
            mission_hud_panel.enabled = show_mission_hud
            mission_hud_border.enabled = show_mission_hud
            info_text.enabled = True
            controls_text.enabled = True
            force_hud.enabled = show_vectors
            separation_slider.enabled = True
            alignment_slider.enabled = True
            cohesion_slider.enabled = True
            waypoint_slider.enabled = True
            separation_slider.text_entity.enabled = True
            alignment_slider.text_entity.enabled = True
            cohesion_slider.text_entity.enabled = True
            waypoint_slider.text_entity.enabled = True
            boundary_faces.enabled = True
            boundary_faces.color = rgb(15, 28, 48, 160) # Restore normal mode opacity
            sky_sphere.enabled = True
            star_field.enabled = True
            floor_plane.enabled = True
            if obs_mode:
                mode_bar_bg.enabled = True
                mov_tag = ' [MOVING]' if obs_moving_mode else ''
                mode_bar.text = (f'[ OBS MODE{mov_tag} ]  Height:{int(obs_height[0])}'
                                 f'  |  M=Moving  ↑↓=Height  LClick=Place  RClick=Undo')
            else:
                mode_bar_bg.enabled = False
                mode_bar.text = ''

        if show_benchmark and not cinematic_mode:
            bench_panel.enabled = True
            bench_text.text = (
                f'FPS: {fps_v}  Sim TPS: {int(_physics_tps[0])}\n'
                f'{swarm.metrics.get_hud_text()}'
            )
        else:
            bench_panel.enabled = False

    if held_keys['r']:
        perform_full_reset()
    
    render_profiler.end_frame()
    render_stats.record_render_frame(time.dt if hasattr(time, 'dt') else 0.016)


def save_metrics_csv():
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

    swarm.metrics.export_csv("parallel_analysis.csv")
    swarm.metrics.print_report()

def input(key):
    global cinematic_mode, obs_mode, obs_moving_mode
    global show_neighbor_lines, show_heatmap, show_trails, show_vectors, show_centroid
    global show_benchmark
    global obs_ghost, obs_height, static_obs_ents, user_added, user_moving_obs, _initial_static_count

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

    if key == '9':
        config.transport_drone_count = min(swarm.num_boids, config.transport_drone_count + 5)
        return
        
    if key == '8':
        config.transport_drone_count = max(1, config.transport_drone_count - 5)
        return

    if key == 'v':
        show_vectors = not show_vectors
        force_hud.enabled = show_vectors

    if key == 'b':
        show_benchmark = not show_benchmark
        bench_panel.enabled = show_benchmark

    if   key == '1': swarm.set_method('naive')
    elif key == '2': swarm.set_method('octree')
    elif key == 'g': show_centroid = not show_centroid
    elif key == 'p': 
        global show_mission_hud
        show_mission_hud = not show_mission_hud
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
            min_len = min(len(hmap_verts), len(hmap_colors))
            if min_len > 0:
                hmap_ent.model.vertices = hmap_verts[:min_len]
                hmap_ent.model.colors = hmap_colors[:min_len]
                try:
                    hmap_ent.model.generate()
                except:
                    pass

    if key == 'c':
        cinematic_mode = not cinematic_mode
        cinematic_mode_running[0] = cinematic_mode
        if cinematic_mode:
            _saved_cam_z[0] = camera.z
            # Neutralize EditorCamera (update block handles the actual monkey-patch)
        else:
            # Restore EditorCamera if monkey-patched
            if hasattr(editor_cam, '_orig_update'):
                editor_cam.update = editor_cam._orig_update
                editor_cam.input = editor_cam._orig_input
            camera.z = _saved_cam_z[0]
    elif key == 'o':
        obs_mode = not obs_mode
        obs_ghost.enabled = False
        if not obs_mode: obs_moving_mode = False

    if obs_mode:
        if key == 'up arrow':
            obs_height[0] = min(obs_height[0] + 30, H - 20)
        elif key == 'down arrow':
            obs_height[0] = max(obs_height[0] - 30, 20)
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
