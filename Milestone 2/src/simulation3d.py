from ursina import *
from flock3d import Flock3D
import numpy as np
import config

# Create Ursina App
app = Ursina(borderless=False)
window.color = color.black
camera.clip_plane_far = 10000

# Pivot point for EditorCamera (center of flock area)
center = Vec3(config.width/2, config.height/2, config.width/2)

# EditorCamera orbits around a pivot
# Right-Click + drag to rotate, scroll to zoom, middle-click to pan
editor_cam = EditorCamera()
editor_cam.position = center
editor_cam.rotation = (30, -30, 0)  # Angled view looking down
editor_cam.target_z = -1500         # Zoom distance from pivot

# Lighting
PointLight(parent=scene, position=center, color=color.white)
DirectionalLight(parent=scene, y=2, z=3, shadows=True)
AmbientLight(parent=scene, color=color.gray)

# Visual Reference - Center of flocking area
Entity(model='cube', scale=50, position=center, color=color.red)

# Visuals
flock = Flock3D()
boid_entities = []
direction_lines = []
show_lines = True

for i in range(flock.num_boids):
    parent = Entity()
    # Visual Body (Pyramid)
    visual = Entity(parent=parent, model=Cone(4), color=color.cyan, 
                    scale=(5, 5, 15), rotation_x=90, texture='white_cube')
    
    # Direction Line (Translucent)
    direction_line = Entity(parent=parent, model='cube', color=color.rgba(255, 255, 255, 100),
                            scale=(0.5, 0.5, 40), position=(0, 0, 20)) 
    
    boid_entities.append(parent)
    direction_lines.append(direction_line)

# Floor plane for spatial reference
Entity(model='plane', scale=(config.width, 1, config.width),
       position=(config.width/2, 0, config.width/2),
       color=color.dark_gray, alpha=0.5)

def update():
    flock.update()
    
    for i, e in enumerate(boid_entities):
        e.position = Vec3(flock.positions[i][0], flock.positions[i][1], flock.positions[i][2])
        target = e.position + Vec3(flock.velocities[i][0], flock.velocities[i][1], flock.velocities[i][2])
        e.look_at(target)

    if held_keys['r']:
        flock.__init__()

def input(key):
    global show_lines
    if key == 'l':
        show_lines = not show_lines
        for line in direction_lines:
            line.enabled = show_lines

app.run()
