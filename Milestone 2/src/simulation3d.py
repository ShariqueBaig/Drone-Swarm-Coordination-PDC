"""
simulation3d.py -- Usman (C3.1-C3.5) | PDC Drone Swarm | Milestone 3

KEYS:
  1 / 2 / 3      Algorithm: Octree | Grid Hash | Naive O(N^2)
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
  Arrow keys     Pan camera: Left/Right = X   Up/Down = Z (forward/back)
  Page Up/Down   Pan camera: Y (vertical)
  Left-Click     Set waypoint (normal mode)
  Right-Click    Clear waypoint (normal mode)
"""
from ursina import *
from ursina.prefabs.slider import ThinSlider
from swarm_3d import SwarmManager3D
from environment3d import Environment3D
import numpy as np
import config, math, random, csv, time

# ── RGB HELPER ────────────────────────────────────────────────────────────────
def rgb(r, g, b, a=255):
    return color.rgba(r/255.0, g/255.0, b/255.0, a/255.0)

# ── APP ───────────────────────────────────────────────────────────────────────
app = Ursina(borderless=False, title='PDC Drone Swarm | M3 | C3.1-C3.5')
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
CAM_PAN = 220           # units / sec for arrow-key pan

# ── LIGHTING ──────────────────────────────────────────────────────────────────
d_light = DirectionalLight(parent=scene, y=8, z=4, color=rgb(160, 185, 255))
d_light.look_at(Vec3(0, -1, 0.4))
AmbientLight(parent=scene, color=rgb(18, 22, 38))

# ── SKYBOX ────────────────────────────────────────────────────────────────────
Entity(model='sphere', scale=15000, color=rgb(4, 6, 13),
       double_sided=True, unlit=True)
random.seed(77)
_sv = [Vec3(random.uniform(-6000,6000), random.uniform(80,4500),
            random.uniform(-6000,6000)) for _ in range(220)]
Entity(model=Mesh(vertices=_sv, mode='point', thickness=3),
       color=rgb(195, 215, 255), unlit=True)

# ── FLOOR ─────────────────────────────────────────────────────────────────────
Entity(model='plane', scale=(W, 1, W), position=(W/2, 0, W/2),
       color=rgb(9, 14, 23), unlit=True)

GRID_DIVS = 14
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
_bed = [(0,1),(1,2),(2,3),(3,0)]   # bottom square
_ted = [(4,5),(5,6),(6,7),(7,4)]   # top square
_pil = [(0,4),(1,5),(2,6),(3,7)]   # vertical corner pillars

for a, b in _bed + _ted + _pil:
    Entity(model=Mesh(vertices=[Vec3(*_cn[a]), Vec3(*_cn[b])], mode='line', thickness=2),
           color=rgb(55, 105, 190, 235), unlit=True)

# ── OBSTACLES ─────────────────────────────────────────────────────────────────
# KEY: env.all_obstacles is a @property returning a NEW list every access.
# Appending to it is silently discarded. We must use env.obstacles (persistent)
# for static/user-placed obstacles and track dynamic entities separately.
_initial_static_count = len(env.obstacles)   # = 2 (pre-defined static only)

static_obs_ents = []   # tracks env.obstacles entities (static + user-placed)
for ob in env.obstacles:
    e = Entity(model='cube', scale=ob[3]*0.85, position=(ob[0],ob[1],ob[2]),
               color=rgb(215, 35, 55), unlit=True, wireframe=True,
               rotation=(35, 35, 0))
    static_obs_ents.append(e)

dyn_obs_ents = []      # tracks env.dynamic_obstacles entities
for d in env.dynamic_obstacles:
    e = Entity(model='cube', scale=d.radius*0.85, position=(d.x,d.y,d.z),
               color=rgb(180, 50, 200), unlit=True, wireframe=True,
               rotation=(35, 35, 0))
    dyn_obs_ents.append(e)

# ── OBSTACLE PLACEMENT ────────────────────────────────────────────────────────
obs_mode        = False
obs_moving_mode = False        # next placement will be a moving obstacle
obs_height      = [60.0]       # current ghost Y
user_added      = []           # list of dicts: {ent, idx, moving}
user_moving_obs = []           # list of dicts for oscillation

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
# Removed decorative centroid_glow for cleaner aesthetics

# Using centroid_marker as the only white prism (dynamic center of formation)
# Removed static global_center_prism to avoid visual clutter

# ── NEIGHBOR LINES ────────────────────────────────────────────────────────────
show_neighbor_lines = False
nb_line_ent = Entity(model=Mesh(vertices=[], mode='line', thickness=1),
                     color=rgb(0, 190, 255, 70), unlit=True, enabled=False)

# ── TRAILS ────────────────────────────────────────────────────────────────────
show_trails = True
show_centroid = False # M3: Global Center Marker Toggle (G key)

# ── HEATMAP ───────────────────────────────────────────────────────────────────
show_heatmap  = False
HMAP_CELL     = 80
hmap_cols     = int(W / HMAP_CELL) + 1
hmap_rows     = int(W / HMAP_CELL) + 1
total_hcells  = hmap_cols * hmap_rows
visited_cells = set()           # always tracked regardless of heatmap toggle

hmap_verts  = []
hmap_colors = []
hmap_ent = Entity(model=Mesh(vertices=[], colors=[], mode='point', thickness=14),
                  unlit=True, enabled=False)

def _stamp_tile(cx, cz):
    x = cx * HMAP_CELL + HMAP_CELL/2
    z = cz * HMAP_CELL + HMAP_CELL/2
    hmap_verts.append(Vec3(x, 0, z))
    hmap_colors.append(rgb(0, 190, 255, 180)) # sleek uniform aesthetic blue point
    
    if hmap_ent.enabled:
        hmap_ent.model.vertices = hmap_verts
        hmap_ent.model.colors = hmap_colors
        hmap_ent.model.generate()

# ── DRONES ────────────────────────────────────────────────────────────────────
boid_entities = []
for i in range(swarm.num_boids):
    drone = Entity(model='sphere', scale=7, color=rgb(0, 210, 255), unlit=True)
    Entity(parent=drone, model=Cone(6),
           scale=(0.45, 0.45, 1.7), position=(0,0,0.65),
           color=rgb(160, 235, 255), unlit=True)
    drone.trail_verts = []
    drone.trail = Entity(model=Mesh(vertices=[], mode='line', thickness=2),
                         color=rgb(0, 145, 220, 185), unlit=True)
    
    # Behavior Glow Ring (Micro-Indicator)
    e_ring = Entity(parent=drone, model='circle', scale=2.0, rotation_x=90, y=-0.5,
                    color=color.clear, unlit=True)
    drone.behavior_ring = e_ring
    
    boid_entities.append(drone)
    
# ── M3 TASK MARKERS & ALLOCATOR PRISMS (C2.3) ────────────────────────────────
task_markers = []
for t_idx, t_pos in enumerate(swarm.tasks):
    # Skip visualizing tasks 4-5 (obstacle centers) to avoid visual overlap
    if t_idx in [4, 5]:
        continue
    
    tm = Entity(position=(t_pos[0], t_pos[1], t_pos[2]), enabled=True)
    
    icon = Entity(parent=tm, model='diamond', scale=(30, 50, 30), color=rgb(0, 255, 255, 180), unlit=True)
    # Removed decorative spinning ring from task marker
    
    ring = Entity(parent=tm, model='circle', scale=25, rotation_x=90, color=rgb(150, 150, 150, 60), unlit=True)
    task_markers.append({'base': tm, 'icon': icon, 'ring': ring})

# ── HUD ───────────────────────────────────────────────────────────────────────
# Consolidate UI into a narrower transparent dashboard
ui_panel = Entity(parent=camera.ui, model='quad', scale=(0.38, 0.95),
                  position=(-0.72, 0.0), color=rgb(8, 12, 22, 60)) # More transparent
Entity(parent=camera.ui, model='quad', scale=(0.385, 0.955),
       position=(-0.72, 0.0), color=rgb(0, 140, 255, 10), z=1) # High Z for subtle glow

info_text = Text(text='Initializing...', position=(-0.89, 0.45),
                 scale=0.65, color=rgb(180, 180, 180), background=False)

# Sliders nested properly
if not hasattr(config, 'waypoint_weight'):
    config.waypoint_weight = 2.5

slider_x = -0.72
slider_start_y = 0.08
def _make_slider(text, val, y_off):
    s = ThinSlider(text=text, dynamic=True, min=0, max=10, default=val,
                   x=slider_x, y=y_off, parent=camera.ui, scale=0.8)
    s.label.color = color.white
    s.label.scale = 1.1
    s.label.position = (-0.1, 0.01) # Push label left of the bar
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
         '1/2/3 : Algo      L : Lines\n'
         'T : Trails        H : Map\n'
         'O : Obs Mode      R : Reset\n'
         'C : Cinematic\n'
         'W/A/S/D : Pan XZ  Q/E: Pan Y\n'
         'L-Click Floor: Set Waypoint\n'
         'R-Click: Clear Waypoint\n'
         'G : Center Toggle\n'
         'P : Export CSV\n'
         '[Obs] Arrow Up/Down: Height\n'
         '[Obs] L-Click: Place  M: Move\n'
         'V : Diagnostic Mode',
    position=(-0.89, -0.21), scale=0.72, color=color.azure
)

show_vectors = False
# Macro Intent Indicator: Cyan diamond showing swarm net force direction (V key diagnostic)
intent_indicator = Entity(model='diamond', scale=(35, 60, 35), color=rgb(0, 210, 255, 120),
                          unlit=True, wireframe=True, enabled=False)
intent_glow = Entity(parent=intent_indicator, model='circle', scale=1.2, rotation_x=90,
                     color=rgb(0, 210, 255, 60), unlit=True)

# ── FORCE BREAKDOWN HUD (C2.4) ───────────────────────────────────────────────
force_hud = Entity(parent=camera.ui, enabled=False)
force_bg = Entity(parent=force_hud, model='quad', scale=(0.22, 0.25), position=(0.83, -0.45), color=rgb(8,12,22,30)) # More transparent
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

_RAD = 0.27
RADAR_POS = (0.78, 0.35)
Entity(parent=camera.ui, model='quad', scale=(_RAD, _RAD), position=RADAR_POS,
       color=rgb(8, 12, 22, 80))
Entity(parent=camera.ui, model='quad', scale=(_RAD+0.007, _RAD+0.007),
       position=RADAR_POS, color=rgb(0, 170, 255, 20), z=0.01)
Text(parent=camera.ui, text='RADAR', origin=(0,0),
     position=(0.78, 0.48), scale=0.75, color=rgb(150, 150, 150))

radar_sweep = Entity(parent=camera.ui,
                     model=Mesh(vertices=[Vec3(0,0,0), Vec3(0,_RAD*0.48,0)],
                                mode='line', thickness=2),
                     color=rgb(0, 255, 110, 170), position=RADAR_POS)
radar_dots = []
for _ in range(swarm.num_boids):
    radar_dots.append(Entity(parent=camera.ui, model='circle',
                              scale=0.008, color=rgb(0, 215, 255)))

# Mode bar (centre top) -- opaque background for visibility
mode_bar_bg = Entity(parent=camera.ui, model='quad', scale=(0.6, 0.06), position=(0, 0.45), color=rgb(8, 12, 22, 140), enabled=False)
mode_bar = Text(text='', origin=(0,0), position=(0, 0.45),
                scale=1.0, color=rgb(255, 165, 30), background=False)

# Volumetric Bounding Box (Context)
volume_outline = Entity(model='cube', scale=(W, H, D), position=(W/2, H/2, D/2), 
                        color=rgb(0, 210, 255, 15), wireframe=True, unlit=True, enabled=True)

# Aesthetic Scanning Plane
scan_plane = Entity(model='quad', scale=(W, D), rotation_y=90, 
                    color=rgb(0, 255, 255, 10), unlit=True, enabled=False)

# ── M3: GHOST DRONE (HOSTILE TARGET) ─────────────────────────────────────────
ghost_drone = Entity(model='diamond', scale=15, color=rgb(255, 0, 80), 
                     unlit=True, enabled=True)
Entity(parent=ghost_drone, model='circle', scale=2.5, rotation_x=90, 
       color=rgb(255, 0, 80, 40), unlit=True)
ghost_label = Text(parent=ghost_drone, text='HOSTILE', y=1.5, scale=20, 
                   color=color.red, origin=(0,0), billboarding=True)

# EMP Pulse Effect (for successful intercept)
emp_pulse = Entity(model='sphere', scale=1, color=rgb(255, 0, 80, 0), unlit=True)

# ── MISSION FLEET HUD (C2.3 - Highlighting/Selection) ────────────────────────
mission_hud_panel = Entity(parent=camera.ui, model='quad', scale=(0.20, 0.75), 
                           position=(0.78, -0.1), color=rgb(8, 12, 22, 40)) # More transparent
Entity(parent=camera.ui, model='quad', scale=(0.205, 0.755),
       position=(0.78, -0.1), color=rgb(0, 140, 255, 8), z=0.01) # Subtle glow
Text(parent=mission_hud_panel, text='FLEET COMMAND', position=(0, 0.46), scale=0.65, color=rgb(150, 150, 150), origin=(0,0))

# M3: Fault Injection Buttons
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

intercept_btn = Button(parent=mission_hud_panel, text='INTERCEPT GHOST', 
                       scale=(0.85, 0.07), x=0.02, y=-0.16, color=rgb(80, 40, 40, 60))
intercept_btn.on_click = lambda: select_mission(4)

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
mission_btns.append(intercept_btn)

coverage_text = Text(parent=mission_hud_panel, text='COVERAGE: 0.0%', 
                     position=(0, 0.42), scale=0.55, color=rgb(150, 150, 150), origin=(0,0))

# Mission Success Banner
mission_banner = Entity(parent=camera.ui, model='quad', scale=(0.8, 0.15), 
                        position=(0, 0), color=rgb(20, 180, 255, 0), enabled=False)
banner_text = Text(parent=mission_banner, text='AREA CLEARED | RECALL INITIATED', 
                   origin=(0,0), scale=2.5, color=color.white)

# ── TIMING & METRICS ─────────────────────────────────────────────────────────
_frame = [0]
last_vis_count = [0] 
metrics_log = []     # Evaluation data storage
log_timer = 0        # Tracks seconds for telemetry

# ── UPDATE ────────────────────────────────────────────────────────────────────
def update():
    swarm.update()
    _frame[0] += 1
    _t = time.time()
    cov_pct = swarm.coverage_pct # Define here for logic visibility

    # Camera keyboard pan (WASD relative to view, QE for Y)
    ep = editor_cam.position
    try:
        y_rot = math.radians(editor_cam.rotation_y)
        fwd = Vec3(math.sin(y_rot), 0, math.cos(y_rot))
        right = Vec3(math.cos(y_rot), 0, -math.sin(y_rot))
    except:
        fwd, right = Vec3(0,0,1), Vec3(1,0,0)

    if held_keys['a']:   editor_cam.position -= right * CAM_PAN * time.dt
    if held_keys['d']:   editor_cam.position += right * CAM_PAN * time.dt
    if held_keys['w']:   editor_cam.position += fwd * CAM_PAN * time.dt
    if held_keys['s']:   editor_cam.position -= fwd * CAM_PAN * time.dt
    if held_keys['e']:   editor_cam.position += Vec3(0, CAM_PAN * time.dt, 0)
    if held_keys['q']:   editor_cam.position -= Vec3(0, CAM_PAN * time.dt, 0)

    # Waypoint pulse
    if waypoint_marker.enabled:
        waypoint_marker.rotation_y += 70 * time.dt
        pulse = 1.0 + 0.38 * math.sin(_t * 5)
        waypoint_ring.scale_x = 3 * pulse
        waypoint_ring.scale_y = 3 * pulse

    # Ghost in obstacle mode (ray-plane intersection for floor preview)
    if obs_mode:
        try:
            # Check if mouse is visible and has valid position
            if hasattr(mouse, 'world_point') and mouse.world_point:
                wp = mouse.world_point
                # Position ghost at plane height (obs_height)
                obs_ghost.enabled = True
                obs_ghost.position = Vec3(wp.x, obs_height[0], wp.z)
            else:
                # Fallback: use camera-based ray-casting
                cam_pos = camera.world_position
                # Simple ray projection: move camera forward and down to plane
                dist_to_plane = obs_height[0] - cam_pos.y
                if abs(dist_to_plane) > 1:
                    # Estimate horizontal position from mouse offset
                    mx = (mouse.x - 0.5) * 10
                    mz = (mouse.y - 0.5) * 10
                    obs_ghost.enabled = True
                    obs_ghost.position = Vec3(cam_pos.x + mx, obs_height[0], cam_pos.z + mz)
                else:
                    obs_ghost.enabled = False
        except:
            obs_ghost.enabled = False
            
        obs_ghost.rotation_y += 55 * time.dt

    # Radar sweep
    radar_sweep.rotation_z -= 85 * time.dt

    # Update moving user-placed obstacles into env.obstacles (persistent list)
    for mo in user_moving_obs:
        nx = mo['ox'] + mo['amp'] * math.sin(mo['freq']  * _t)
        nz = mo['oz'] + mo['amp'] * math.cos(mo['freq2'] * _t)
        nx = max(50, min(W-50, nx))
        nz = max(50, min(W-50, nz))
        sidx = mo['static_idx']
        if sidx < len(swarm.env.obstacles):
            old = swarm.env.obstacles[sidx]
            swarm.env.obstacles[sidx] = (nx, old[1], nz, old[3])

    # Sync static + user-placed obstacles (env.obstacles is a real persistent list)
    for i, ob in enumerate(swarm.env.obstacles):
        if i < len(static_obs_ents):
            static_obs_ents[i].position = (ob[0], ob[1], ob[2])
            static_obs_ents[i].rotation_y += 28 * time.dt
    # Sync dynamic obstacles separately (avoids index-mapping confusion)
    for i, d in enumerate(swarm.env.dynamic_obstacles):
        if i < len(dyn_obs_ents):
            dyn_obs_ents[i].position = (d.x, d.y, d.z)
            dyn_obs_ents[i].rotation_y += 28 * time.dt

    # Per-drone update
    active_count = 0
    centroid = Vec3(0, 0, 0)
    do_hmap  = show_heatmap and (_frame[0] % 4 == 0)
    do_trail = show_trails  and (_frame[0] % 4 == 0)
    do_nl    = show_neighbor_lines and (_frame[0] % 5 == 0)

    for i, e in enumerate(boid_entities):
        pos = swarm.positions[i]

        if swarm.dead_mask[i]:
            if e.color != color.red:
                e.color = rgb(255, 30, 30)
                for ch in e.children:
                    if hasattr(ch, 'model') and ch.model:
                        ch.color = rgb(255, 100, 90)
                radar_dots[i].color = rgb(255, 30, 30)
            e.y = max(-15, e.y - 140*time.dt)
            e.rotation_x += 75*time.dt
            continue

        active_count += 1
        p3 = Vec3(pos[0], pos[1], pos[2])
        e.position = p3
        centroid += p3

        vel = swarm.velocities[i]
        spd = math.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
        
        # FLEET HIGHLIGHTING & FORCE DEBUG (TASK C2.4 / C2.3)
        h_id = highlighted_mission[0]
        
        if show_vectors:
            # Behavior Glow (Micro-Indicator)
            forces = [
                (np.linalg.norm(swarm.last_sep[i]), rgb(255, 60, 60)),     # Red
                (np.linalg.norm(swarm.last_aln[i]), rgb(255, 255, 60)),    # Yellow
                (np.linalg.norm(swarm.last_coh[i]), rgb(60, 255, 60)),     # Green
                (np.linalg.norm(swarm.last_waypoint[i]), rgb(60, 210, 255)) # Azure
            ]
            mag_max, col_max = max(forces, key=lambda x: x[0])
            if mag_max > 0.08:
                # Color slicing fix: Color is a Vec4/tuple, use individual attributes
                e.behavior_ring.color = color.rgba(col_max.r, col_max.g, col_max.b, min(0.4, mag_max*0.5))
            else:
                e.behavior_ring.color = color.clear
            
            # Keep original colors in vector mode for fleet context
            e.color = rgb(0, 210, 255)
            e.scale = 7
        else:
            e.behavior_ring.color = color.clear
            if h_id != -1:
                if swarm.mission_type[i] == h_id:
                    e.color = color.white
                    e.scale = 10
                    e.unlit = True
                else:
                    e.color = rgb(40, 45, 50, 150)
                    e.scale = 6
                    e.unlit = False
            else:
                e.color = rgb(0, 210, 255)
                e.scale = 7
                e.unlit = False

        if spd > 0.5:
            e.look_at(p3 + Vec3(vel[0], vel[1], vel[2]))
        
        if hasattr(e, 'children') and len(e.children) > 0:
            e.children[0].color = rgb(160, 235, 255)

        # Trail
        if do_trail:
            e.trail_verts.append(p3)
            if len(e.trail_verts) > 9: e.trail_verts.pop(0)
            if len(e.trail_verts) >= 2:
                e.trail.model.vertices = e.trail_verts
                e.trail.model.generate()

        # Radar dot
        rx = max(-0.13, min(0.13, (pos[0]/W - 0.5) * _RAD))
        rz = max(-0.13, min(0.13, (pos[2]/W - 0.5) * _RAD))
        radar_dots[i].position = Vec3(0.78 + rx, 0.35 - rz, -0.01)

    # Neighbour lines
    if do_nl:
        pi = swarm._last_pairs_i; pj = swarm._last_pairs_j
        if len(pi) > 0:
            nv = []
            for a, b in zip(pi[:180], pj[:180]):
                pa = swarm.positions[a]; pb = swarm.positions[b]
                nv += [Vec3(pa[0],pa[1],pa[2]), Vec3(pb[0],pb[1],pb[2])]
            nb_line_ent.model.vertices = nv
            nb_line_ent.model.generate()
        else:
            nb_line_ent.model.vertices = []
            nb_line_ent.model.generate()

    if active_count > 0:
        centroid /= active_count

    # Cinematic -- tight orbit, close to drones, always looking at centroid
    if cinematic_mode and active_count > 0:
        orbit_r = 260
        angle = _t * 0.32
        cx_c = centroid.x + orbit_r * math.sin(angle)
        cy_c = centroid.y + 100
        cz_c = centroid.z + orbit_r * math.cos(angle)
        editor_cam.position = lerp(editor_cam.position,
                                   Vec3(cx_c, cy_c, cz_c), 4 * time.dt)
        editor_cam.look_at(centroid)

    # M2 Task Visualization (C2.3)
    assigned_count = np.zeros(len(swarm.tasks))
    for tid in swarm.assigned_tasks:
        if tid != -1: assigned_count[tid] += 1
    
    # Update Action For Task Markers
    for tid, tm_dict in enumerate(task_markers):
        if assigned_count[tid] > 0:
            tm_dict['icon'].color = rgb(0, 255, 255, 200)
            tm_dict['icon'].rotation_y += 100 * time.dt
            tm_dict['icon'].scale = Vec3(30, 50, 30) * (1.1 + 0.1 * math.sin(_t*5))
            tm_dict['ring'].color = rgb(255, 200, 0, 150)
        else:
            tm_dict['icon'].color = rgb(150, 150, 150, 150) # Grey = Unassigned
            tm_dict['ring'].color = rgb(150, 150, 150, 60)
            tm_dict['icon'].rotation_y += 20 * time.dt

    # Update Action For HUD Force Bar & Macro Swarm Intent (C2.4)
    if show_vectors and active_count > 0:
        if not hasattr(swarm, 'last_sep'): return # safety check
        
        alive_mask = ~swarm.dead_mask
        l_sep = swarm.last_sep[alive_mask]
        l_aln = swarm.last_aln[alive_mask]
        l_coh = swarm.last_coh[alive_mask]
        l_tsk = swarm.last_waypoint[alive_mask]

        avg_sep = np.mean(np.linalg.norm(l_sep, axis=1))
        avg_aln = np.mean(np.linalg.norm(l_aln, axis=1))
        avg_coh = np.mean(np.linalg.norm(l_coh, axis=1))
        avg_tsk = np.mean(np.linalg.norm(l_tsk, axis=1))
        
        # Drive the labeled bars
        scale_fac = 0.5 
        bar_sep.scale_x = min((avg_sep / scale_fac) * 0.12, 0.12)
        bar_aln.scale_x = min((avg_aln / scale_fac) * 0.12, 0.12)
        bar_coh.scale_x = min((avg_coh / scale_fac) * 0.12, 0.12)
        bar_tsk.scale_x = min((avg_tsk / scale_fac) * 0.12, 0.12)

        # Macro Intent Indicator: Average of all steering forces
        net_force_v = (np.mean(l_sep, axis=0) + np.mean(l_aln, axis=0) + 
                       np.mean(l_coh, axis=0) + np.mean(l_tsk, axis=0))
        
        intent_indicator.enabled = True
        intent_indicator.position = centroid
        intent_indicator.scale = Vec3(35, 60, 35) * (1.0 + 0.2 * math.sin(_t*6))
        
        if np.linalg.norm(net_force_v) > 0.01:
            intent_indicator.look_at(centroid + Vec3(net_force_v[0], net_force_v[1], net_force_v[2]))
            intent_indicator.rotation_x = 90 # Orient diamond correctly after look_at
    else:
        intent_indicator.enabled = False

    # Update Mission Button highlight state
    # Button indices: 0-3: Missions 0-3, 4: Coverage (mission 6), 5: Intercept (mission 4)
    btn_mission_map = [0, 1, 2, 3, 6, 4]  # Maps button index to mission type
    for i, btn in enumerate(mission_btns):
        mission_type = btn_mission_map[i] if i < len(btn_mission_map) else i
        btn.color = rgb(100, 150, 180, 100) if highlighted_mission[0] == mission_type else (rgb(60, 80, 60, 50) if i == 4 else rgb(60, 60, 60, 50))

    # M3: Update Area Coverage Tracking & Telemetry (Reads from Brain)
    if _frame[0] % 2 == 0:
        global log_timer
        # Read coverage status from the Swarm Manager logic
        current_vis = np.sum(swarm.visited_grid)
        cov_pct = (current_vis / (swarm.grid_res**3)) * 100
        coverage_text.text = f'COVERAGE: {cov_pct:.1f}%'

        # Telemetry Snapshot (every ~1 second)
        # M3: Update Area Coverage Tracking & Telemetry (Optimized Brain-Link)
        if _frame[0] % 2 == 0:
            global log_timer
            # cov_pct already updated at top of loop
            coverage_text.text = f'COVERAGE: {cov_pct:.1f}%'

            # Discovery Visuals (Cyan Pulse on new voxels)
            if show_heatmap or show_vectors: # Pulse when heatmap OR diagnostics are ON
                new_voxels = np.argwhere(swarm.visited_grid & ~swarm.last_grid)
                if len(new_voxels) > 0:
                    voxel_size = np.array([W/swarm.grid_res, H/swarm.grid_res, D/swarm.grid_res])
                    for v in new_voxels:
                        pos = v * voxel_size + (voxel_size/2)
                        
                        # Add to persistent heatmap if enabled
                        if show_heatmap:
                            hmap_verts.append(Vec3(*pos))
                            hmap_colors.append(rgb(0, 210, 255, 40)) 
                        
                        # High-Visual Discovery Pulse (Capped for performance)
                        if len(hmap_verts) % 5 == 0: # Only pulse occasionally
                            DiscoveryPulse(pos)
                    
                    if show_heatmap:
                        hmap_ent.model.vertices = hmap_verts
                        hmap_ent.model.colors = hmap_colors
                        hmap_ent.model.generate()
                    
                    # Update Scanning Plane position relative to latest search
                    scan_plane.enabled = True
                    scan_plane.y = (H/2) + math.sin(_t*0.5) * (H/2)
                    scan_plane.color = rgb(0, 210, 255, 10 + 10*math.sin(_t*8))
            else:
                scan_plane.enabled = False

            # Telemetry Snapshot (every ~1 second)
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

    # Formation Centroid & Global Center Viz (C2.2)
    if active_count > 0:
        centroid_marker.enabled = show_centroid
        centroid_marker.position = centroid
        
        # Update Ghost Drone movement (Orbital path)
        _gt = _t * 0.5
        gx = W/2 + math.sin(_gt) * W*0.3
        gz = D/2 + math.cos(_gt*1.3) * D*0.3
        gy = H/2 + math.sin(_gt*0.7) * H*0.2
        ghost_drone.position = (gx, gy, gz)
        
        # Intercept Pulse logic
        intercepting = np.where(~swarm.dead_mask & (swarm.mission_type == 4))[0]
        if len(intercepting) > 0:
            d_to_g = np.linalg.norm(swarm.positions[intercepting] - np.array(ghost_drone.position), axis=1)
            if np.any(d_to_g < 60): # Slightly larger trigger for visual emp
                emp_pulse.position = ghost_drone.position
                emp_pulse.scale_x += 300 * time.dt
                emp_pulse.scale_y = emp_pulse.scale_x
                emp_pulse.scale_z = emp_pulse.scale_x
                if emp_pulse.scale_x > 180: emp_pulse.scale = (1,1,1)
                emp_pulse.color = rgb(255, 0, 80, max(0, 160 - (emp_pulse.scale_x/180)*160))
            else:
                emp_pulse.color = rgb(255, 0, 80, 0); emp_pulse.scale = 1
        else:
            emp_pulse.color = rgb(255, 0, 80, 0); emp_pulse.scale = 1

        # Update swarm knowledge of ghost
        if len(swarm.tasks) > 8:
            swarm.tasks[8] = (gx, gy, gz)
        
        centroid_marker.rotation_y += 45 * time.dt
    else:
        centroid_marker.enabled = False

    # HUD update
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

    vec_s = 'ON' if show_vectors else 'OFF'
    info_text.text = (
        'SWARM  COVERAGE  METRICS\n'
        '------------------------\n'
        f'Active : {fault}{active_count}/{swarm.num_boids}\n'
        f'Dead   : {dead_n}\n'
        f'Cover  : {cov:.1f}%\n'
        f'Status : {wp_s}\n'
        f'Algo   : {algo}\n'
        f'\n'
        f'TOGGLES (Keys)\n'
        f'------------------------\n'
        f'T: Trails {trail_s}\n'
        f'L: Neighbor Lines {nl_s}\n'
        f'H: Coverage Map {hm_s}\n'
        f'G: Center {("ON" if show_centroid else "OFF")}\n'
        f'V: Diagnostics {vec_s}\n'
        f'C: Cinematic {cam_s}\n'
        f'1/2/3: Algorithm\n'
        f'O: Obstacle Mode\n'
        f'R: Reset\n'
        f'FPS    : {fps_v}'
    )

    if obs_mode:
        mode_bar_bg.enabled = True
        mov_tag = ' [MOVING]' if obs_moving_mode else ''
        mode_bar.text = (f'[ OBS MODE{mov_tag} ]  Height:{int(obs_height[0])}'
                         f'  |  M=Moving  ↑↓=Height  LClick=Place  RClick=Undo')
    else:
        mode_bar_bg.enabled = False
        mode_bar.text = ''

    # Reset
    if held_keys['r']:
        swarm.__init__(env)
        waypoint_marker.enabled = False
        swarm.env.target_waypoint = None
        hmap_verts.clear(); hmap_colors.clear()
        hmap_ent.model.vertices = []; hmap_ent.model.generate()
        
        # M4: Reset Telemetry & Success State
        metrics_log.clear()
        log_timer = 0
        mission_banner.enabled = False
        mission_banner.color = rgb(20, 180, 255, 0)
        
        for e in boid_entities:
            e.color = rgb(0, 210, 255); e.y = 0; e.rotation_x = 0
            e.trail_verts.clear()
            e.trail.model.vertices = []; e.trail.model.generate()
        for d in radar_dots: d.color = rgb(0, 215, 255)
        while user_added:
            rec = user_added.pop()
            destroy(rec['ent'])
            if len(swarm.env.obstacles) > _initial_static_count:
                swarm.env.obstacles.pop()
            if static_obs_ents: static_obs_ents.pop()
        user_moving_obs.clear()


def save_metrics_csv():
    """M4: Final Evaluation Export (Replaces old file)"""
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

def input(key):
    global cinematic_mode, obs_mode, obs_moving_mode
    global show_neighbor_lines, show_heatmap, show_trails, show_vectors, show_centroid
    global obs_ghost, obs_height, static_obs_ents, user_added, user_moving_obs, _initial_static_count

    # Debug Force Vectors
    if key == 'v':
        show_vectors = not show_vectors
        force_hud.enabled = show_vectors
        # intent_indicator.enabled = show_vectors  # Removed for cleaner diagnostics
        print(f"[UI] Diagnostic Mode: {'ON' if show_vectors else 'OFF'}")

    # Algorithm switch
    if   key == '1': swarm.set_method('octree')
    elif key == '2': swarm.set_method('grid')
    elif key == '3': swarm.set_method('naive')
    
    # Global center toggle
    elif key == 'g':
        show_centroid = not show_centroid
        print(f"[UI] Global Center: {'ON' if show_centroid else 'OFF'}")

    # Export Evaluation metrics
    elif key == 'p':
        save_metrics_csv()

    # Neighbour lines
    elif key == 'l':
        show_neighbor_lines = not show_neighbor_lines
        nb_line_ent.enabled = show_neighbor_lines
        if not show_neighbor_lines:
            nb_line_ent.model.vertices = []; nb_line_ent.model.generate()

    # Trails
    elif key == 't':
        show_trails = not show_trails
        for e in boid_entities:
            e.trail.enabled = show_trails
            if not show_trails:
                e.trail.model.vertices = []; e.trail.model.generate()
                e.trail_verts.clear()

    # Heatmap toggle
    elif key == 'h':
        show_heatmap = not show_heatmap
        hmap_ent.enabled = show_heatmap
        if show_heatmap:
            hmap_ent.model.vertices = hmap_verts
            hmap_ent.model.colors = hmap_colors
            hmap_ent.model.generate()

    # Cinematic
    elif key == 'c':
        cinematic_mode = not cinematic_mode

    # Obstacle mode
    elif key == 'o':
        obs_mode = not obs_mode
        obs_ghost.enabled = False
        if not obs_mode: 
            obs_moving_mode = False
        else:
            print("[Obs Mode] Arrow Up/Down to adjust height, L-Click to place, R-Click to remove")

    # -- Obstacle mode sub-controls --
    if obs_mode:
        # Height control with arrow keys (more intuitive)
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
            # Store in env.obstacles (persistent list, NOT all_obstacles property)
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

    # ── Normal mode (NOT obstacle mode) ──────────────────────────────────────
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
