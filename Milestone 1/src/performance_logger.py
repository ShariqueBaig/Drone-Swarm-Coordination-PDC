import csv
import time
import psutil
import os
import numpy as np  # Missing import!
import config       # Missing import!
from datetime import datetime

class PerformanceLogger:
    """Logs performance metrics for benchmarking"""
    
    def __init__(self, log_file="optimized_benchmark.csv"):
        self.log_file = log_file
        self.start_time = None
        self.frame_count = 0
        self.frame_times = []  # Store last 60 frame times for accurate FPS
        
        # Fresh log each session
        with open(self.log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'timestamp', 'frame', 'fps', 'cpu_percent', 
                'memory_mb', 'num_drones', 'avg_neighbors', 'method'
            ])
    
    def start_frame(self):
        """Call at beginning of each frame"""
        self.start_time = time.time()
        
    def end_frame(self, swarm_manager):
        """Call at end of each frame to log metrics"""
        end_time = time.time()
        frame_time = end_time - self.start_time
        self.frame_times.append(frame_time)
        self.frame_count += 1
        
        # Keep only last 60 frame times (approx 1 second at 60fps)
        if len(self.frame_times) > 60:
            self.frame_times.pop(0)
        
        # Log every 5 frames for more responsive graphs
        if self.frame_count % 5 == 0:
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            avg_neighbors = getattr(swarm_manager, 'avg_neighbors', 0)
            method = getattr(swarm_manager, 'use_method', 'unknown')
            
            # Log to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.frame_count,
                    round(fps, 2),
                    round(cpu_percent, 1),
                    round(memory, 1),
                    len(swarm_manager.positions),
                    round(avg_neighbors, 2),
                    method
                ])
            
            # Optional console print
            if self.frame_count % 100 == 0:
                print(f"[PERF] Frame {self.frame_count}: {fps:.1f} FPS | Mem: {memory:.1f}MB | Method: {method}")