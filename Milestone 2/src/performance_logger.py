import csv
import time
import psutil
import os
from datetime import datetime


class PerformanceLogger:
    """Logs performance metrics — buffered to avoid disk I/O in hot path."""

    def __init__(self, log_file="optimized_benchmark.csv"):
        self.log_file = log_file
        self.start_time = None
        self.frame_count = 0
        self.frame_times = []
        self._buffer = []          # in-memory buffer; flushed every 100 frames

        with open(self.log_file, 'w', newline='') as f:
            csv.writer(f).writerow([
                'timestamp', 'frame', 'fps', 'cpu_percent',
                'memory_mb', 'num_drones', 'avg_neighbors', 'method'
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
            cpu = psutil.cpu_percent()
            mem = psutil.Process().memory_info().rss / 1024 / 1024
            avg_nb = getattr(swarm_manager, 'avg_neighbors', 0)
            method = getattr(swarm_manager, 'use_method', 'unknown')

            self._buffer.append([
                datetime.now().isoformat(),
                self.frame_count,
                round(fps, 2),
                round(cpu, 1),
                round(mem, 1),
                len(swarm_manager.positions),
                round(avg_nb, 2),
                method
            ])

            # Flush buffer to disk every 100 frames (not every 5)
            if self.frame_count % 100 == 0:
                with open(self.log_file, 'a', newline='') as f:
                    w = csv.writer(f)
                    w.writerows(self._buffer)
                self._buffer.clear()
                print(f"[PERF] Frame {self.frame_count}: {fps:.1f} FPS | "
                      f"Mem: {mem:.1f}MB | Method: {method}")

    def flush(self):
        """Flush remaining buffer on exit."""
        if self._buffer:
            with open(self.log_file, 'a', newline='') as f:
                csv.writer(f).writerows(self._buffer)
            self._buffer.clear()