import csv
import time
import psutil
import os
import numpy as np  # Missing import!
import config       # Missing import!
from datetime import datetime

class PerformanceLogger:
    """Logs performance metrics for benchmarking"""
    
    def __init__(self, log_file="benchmark_log.csv"):
        self.log_file = log_file
        self.start_time = None
        self.frame_count = 0
        self.frame_times = []  # Store last 100 frame times for accurate FPS
        
        # Create CSV with headers
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
        
        # Keep only last 100 frame times
        if len(self.frame_times) > 100:
            self.frame_times.pop(0)
        
        # Log every 100 frames
        if self.frame_count % 100 == 0:
            # Calculate average FPS over last 100 frames
            avg_frame_time = sum(self.frame_times) / len(self.frame_times)
            fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
            
            # System metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Swarm metrics - use the pre-computed avg_neighbors from swarm_manager
            # This avoids O(n²) calculation again
            avg_neighbors = swarm_manager.avg_neighbors
            
            # Get current method
            method = getattr(swarm_manager, 'use_method', 'unknown')
            
            # Log to CSV
            with open(self.log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.frame_count,
                    f"{fps:.2f}",
                    f"{cpu_percent:.1f}",
                    f"{memory:.1f}",
                    len(swarm_manager.positions),
                    f"{avg_neighbors:.2f}",
                    method
                ])
            
            # Also print to console
            print(f"\n[PERF] Frame {self.frame_count}: {fps:.1f} FPS | "
                  f"CPU: {cpu_percent:.1f}% | Mem: {memory:.1f}MB | "
                  f"Avg Neighbors: {avg_neighbors:.2f} | Method: {method}")