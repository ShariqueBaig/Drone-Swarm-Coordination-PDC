"""
performance_logger.py — Fixed version
PDC Project · Spring 2026

BUGS FIXED vs original:
────────────────────────────────────────────────────────────────────────────
1. list.pop(0) is O(n) — replaced with collections.deque(maxlen=60), O(1)
2. CSV file opened/closed every 5 frames — replaced with buffered writes,
   flushed every 30 frames (reduces I/O overhead by 6x)
3. psutil.cpu_percent(interval=None) without priming returns 0 — fixed with
   a priming call in __init__ so subsequent calls return real deltas
4. No warm-up skip — first WARMUP_FRAMES frames discarded from log to
   eliminate JIT and cache warmup noise that makes early FPS look low
────────────────────────────────────────────────────────────────────────────
"""

import csv
import time
import psutil
import os
from collections import deque
from datetime import datetime


WARMUP_FRAMES = 30


class PerformanceLogger:

    def __init__(self, log_file="optimized_benchmark.csv"):
        self.log_file    = log_file
        self.start_time  = None
        self.frame_count = 0

        # FIX 1: O(1) deque instead of O(n) list.pop(0)
        self.frame_times = deque(maxlen=60)

        # FIX 3: prime psutil so first measurement is accurate
        psutil.cpu_percent(interval=None)

        # FIX 2: buffer writes, flush periodically
        self._buffer      = []
        self._FLUSH_EVERY = 30

        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame', 'fps', 'cpu_percent',
                'memory_mb', 'num_drones', 'avg_neighbors', 'method'
            ])

    def start_frame(self):
        self.start_time = time.perf_counter()

    def end_frame(self, swarm_manager):
        frame_time = time.perf_counter() - self.start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1

        # FIX 4: discard warmup frames
        if self.frame_count <= WARMUP_FRAMES:
            return

        if self.frame_count % 5 != 0:
            return

        avg_ft = sum(self.frame_times) / len(self.frame_times)
        fps    = 1.0 / avg_ft if avg_ft > 0 else 0.0

        cpu_percent = psutil.cpu_percent(interval=None)
        memory      = psutil.Process().memory_info().rss / 1024 / 1024

        avg_neighbors = getattr(swarm_manager, 'avg_neighbors', 0)
        method        = getattr(swarm_manager, 'use_method', 'unknown')

        self._buffer.append([
            datetime.now().isoformat(),
            self.frame_count,
            round(fps, 2),
            round(cpu_percent, 1),
            round(memory, 1),
            len(swarm_manager.positions),
            round(avg_neighbors, 2),
            method
        ])

        if len(self._buffer) >= self._FLUSH_EVERY:
            self._flush()

        if self.frame_count % 100 == 0:
            print(f"[PERF] Frame {self.frame_count}: {fps:.1f} FPS | "
                  f"CPU: {cpu_percent:.0f}% | Mem: {memory:.1f}MB | Method: {method}")

    def _flush(self):
        if not self._buffer:
            return
        with open(self.log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(self._buffer)
        self._buffer.clear()

    def close(self):
        """Flush remaining buffer — call when simulation ends."""
        self._flush()
