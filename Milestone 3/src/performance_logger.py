"""
performance_logger.py — Performance Metric Logging · Milestone 3

═══ PDC TECHNIQUE: Amdahl's Law (Timing Instrumentation) ═══
Logs per-frame timing breakdown, parallel speedup ratio, and
serial fraction estimates for Amdahl's/Gustafson's analysis.
"""

import csv
import time
import os
from datetime import datetime

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


class PerformanceLogger:
    """Logs performance metrics — buffered to avoid disk I/O in hot path.

    Milestone 3: Added per-technique timing breakdown and parallel speedup.
    """

    def __init__(self, log_file="optimized_benchmark.csv"):
        self.log_file = log_file
        self.start_time = None
        self.frame_count = 0
        self.frame_times = []
        self._buffer = []

        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'frame', 'fps', 'cpu_percent',
                'memory_mb', 'num_drones', 'avg_neighbors', 'method',
                'serial_fraction', 'amdahl_4core', 'gpu_active'
            ])

    def start_frame(self):
        self.start_time = time.perf_counter()

    def end_frame(self, swarm_manager):
        frame_time = time.perf_counter() - self.start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1

        if len(self.frame_times) > 60:
            self.frame_times.pop(0)

        if self.frame_count % 5 == 0:
            avg_ft = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_ft if avg_ft > 0 else 0.0
            cpu = psutil.cpu_percent() if HAS_PSUTIL else 0.0
            mem = psutil.Process().memory_info().rss / 1024 / 1024 if HAS_PSUTIL else 0.0
            avg_nb = getattr(swarm_manager, 'avg_neighbors', 0)
            method = getattr(swarm_manager, 'use_method', 'unknown')

            # Milestone 3: Extract parallel metrics
            serial_frac = 0.0
            amdahl_4 = 1.0
            gpu_active = False
            if hasattr(swarm_manager, 'metrics'):
                serial_frac = swarm_manager.metrics.get_serial_fraction()
                amdahl_4 = swarm_manager.metrics.amdahl_speedup(4)
            try:
                from swarm_3d import GPU_AVAILABLE
                gpu_active = GPU_AVAILABLE
            except ImportError:
                pass

            self._buffer.append([
                datetime.now().isoformat(),
                self.frame_count,
                round(fps, 2),
                round(cpu, 1),
                round(mem, 1),
                len(swarm_manager.positions),
                round(avg_nb, 2),
                method,
                round(serial_frac, 4),
                round(amdahl_4, 2),
                gpu_active
            ])

            if self.frame_count % 100 == 0:
                with open(self.log_file, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(self._buffer)
                self._buffer.clear()
                print(f"[PERF] Frame {self.frame_count}: {fps:.1f} FPS | "
                      f"Mem: {mem:.1f}MB | Method: {method} | "
                      f"Serial: {serial_frac:.1%} | Amdahl(4): {amdahl_4:.2f}x")

    def flush(self):
        """Flush remaining buffer on exit."""
        if self._buffer:
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerows(self._buffer)
            self._buffer.clear()
