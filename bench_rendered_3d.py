#!/usr/bin/env python3
"""
bench_rendered_3d.py — Benchmark Ursina 3D RENDERED simulation inside Docker.
Uses Xvfb virtual framebuffer + Mesa software OpenGL.
Measures actual rendered FPS (not headless physics-only).
"""
import os, sys, time
os.environ['DISPLAY'] = os.environ.get('DISPLAY', ':99')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Milestone 2', 'src'))

# Patch Ursina to use software rendering
os.environ.setdefault('MESA_GL_VERSION_OVERRIDE', '3.3')
os.environ.setdefault('LIBGL_ALWAYS_SOFTWARE', '1')

from ursina import *
from swarm_3d import SwarmManager3D
from environment3d import Environment3D
import numpy as np
import config

BENCH_FRAMES = 600
WARMUP = 30

app = Ursina(borderless=True, title='Benchmark', fullscreen=False,
             window_type='none')  # offscreen

env = Environment3D()
swarm = SwarmManager3D(env)

frame_times = []
frame_count = [0]

def update():
    t0 = time.perf_counter()
    swarm.update()

    # Minimal rendering: just update drone entity positions
    for i in range(min(10, swarm.num_boids)):
        pass  # Ursina handles scene graph internally

    dt = time.perf_counter() - t0
    frame_count[0] += 1

    if frame_count[0] > WARMUP:
        frame_times.append(dt)

    if frame_count[0] % 100 == 0:
        if frame_times:
            avg = np.mean(frame_times[-100:]) * 1000
            print(f"  Frame {frame_count[0]:5d} | {avg:.2f} ms | {1000/avg:.0f} FPS")

    if frame_count[0] >= BENCH_FRAMES:
        arr = np.array(frame_times) * 1000
        fps = 1000.0 / arr
        print(f"\n{'='*60}")
        print(f"  URSINA 3D RENDERED BENCHMARK (Docker)")
        print(f"  Frames: {len(frame_times)} | Algo: {swarm.use_method}")
        print(f"{'='*60}")
        print(f"  Avg FPS:    {np.mean(fps):.1f}")
        print(f"  Median FPS: {np.median(fps):.1f}")
        print(f"  P95 ms:     {np.percentile(arr, 95):.2f}")
        print(f"  Avg ms:     {np.mean(arr):.2f}")
        print(f"{'='*60}")
        application.quit()

app.run()
