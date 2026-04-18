"""
benchmark_container.py - Quantifiable performance testing for containerized drone swarm (CPU-Only)
Measures compute time and system metrics before and after containerization
"""

import time
import numpy as np
import psutil
import csv
import json
import os
from datetime import datetime
from pathlib import Path

class PerformanceBenchmark:
    def __init__(self, output_dir="benchmarks"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "fps_readings": [],
            "cpu_usage": [],
            "memory_usage": [],
            "frame_times": [],
            "compute_times": []
        }
        self.frame_times_buffer = []
        self.compute_times_buffer = []
        
    def get_system_metrics(self):
        """Capture current system performance metrics"""
        try:
            cpu = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            
            return {
                "cpu": cpu,
                "memory": memory.percent,
                "memory_mb": memory.used / (1024**2),
            }
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return None
    
    def record_frame(self, frame_time, compute_time):
        """Record frame and compute time"""
        self.frame_times_buffer.append(frame_time)
        self.compute_times_buffer.append(compute_time)
        
        if len(self.frame_times_buffer) % 100 == 0:
            # Log metrics every 100 frames
            fps = 1.0 / np.mean(self.frame_times_buffer[-100:]) if self.frame_times_buffer else 0
            self.metrics["fps_readings"].append(fps)
            
            metrics = self.get_system_metrics()
            if metrics:
                self.metrics["cpu_usage"].append(metrics["cpu"])
                self.metrics["memory_usage"].append(metrics["memory"])
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        self.metrics["end_time"] = datetime.now().isoformat()
        
        if not self.frame_times_buffer:
            print("No frame data collected")
            return
        
        # Calculate statistics
        frame_times = np.array(self.frame_times_buffer)
        compute_times = np.array(self.compute_times_buffer)
        
        stats = {
            "duration_seconds": sum(frame_times),
            "total_frames": len(frame_times),
            "fps_avg": 1.0 / np.mean(frame_times),
            "fps_min": 1.0 / np.max(frame_times),
            "fps_max": 1.0 / np.min(frame_times),
            "fps_std": np.std([1.0 / ft for ft in frame_times]),
            "frame_time_avg_ms": np.mean(frame_times) * 1000,
            "frame_time_min_ms": np.min(frame_times) * 1000,
            "frame_time_max_ms": np.max(frame_times) * 1000,
            "frame_time_std_ms": np.std(frame_times) * 1000,
            "compute_time_avg_ms": np.mean(compute_times) * 1000,
            "compute_time_min_ms": np.min(compute_times) * 1000,
            "compute_time_max_ms": np.max(compute_times) * 1000,
            "cpu_avg": np.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "cpu_peak": np.max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "memory_avg_mb": np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "memory_peak_mb": np.max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
        }
        
        # Save JSON report
        report_file = self.output_dir / f"benchmark_{self.timestamp}.json"
        with open(report_file, "w") as f:
            json.dump({**self.metrics, **stats}, f, indent=2, default=str)
        
        # Save CSV for easy analysis
        csv_file = self.output_dir / f"benchmark_{self.timestamp}.csv"
        with open(csv_file, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
        # Print summary
        print("\n" + "="*70)
        print("DRONE SWARM CONTAINERIZED PERFORMANCE BENCHMARK (CPU-ONLY)")
        print("="*70)
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Total Frames: {stats['total_frames']}")
        print(f"\nFPS STATISTICS:")
        print(f"  Average FPS:  {stats['fps_avg']:.2f}")
        print(f"  Min FPS:      {stats['fps_min']:.2f}")
        print(f"  Max FPS:      {stats['fps_max']:.2f}")
        print(f"  Std Dev:      {stats['fps_std']:.2f}")
        print(f"\nFRAME TIME STATISTICS (ms):")
        print(f"  Average:      {stats['frame_time_avg_ms']:.3f} ms")
        print(f"  Min:          {stats['frame_time_min_ms']:.3f} ms")
        print(f"  Max:          {stats['frame_time_max_ms']:.3f} ms")
        print(f"  Std Dev:      {stats['frame_time_std_ms']:.3f} ms")
        print(f"\nCOMPUTE TIME STATISTICS (ms):")
        print(f"  Average:      {stats['compute_time_avg_ms']:.3f} ms")
        print(f"  Min:          {stats['compute_time_min_ms']:.3f} ms")
        print(f"  Max:          {stats['compute_time_max_ms']:.3f} ms")
        print(f"\nRESOURCE USAGE:")
        print(f"  CPU Average:  {stats['cpu_avg']:.1f}%")
        print(f"  CPU Peak:     {stats['cpu_peak']:.1f}%")
        print(f"  Memory Avg:   {stats['memory_avg_mb']:.1f} MiB")
        print(f"  Memory Peak:  {stats['memory_peak_mb']:.1f} MiB")
        print("="*70)
        print(f"Reports saved to: {report_file} and {csv_file}")
        print("="*70 + "\n")
        
        return stats

# Example usage in simulation loop:
# benchmark = PerformanceBenchmark()
# for frame in simulation:
#     frame_start = time.time()
#     compute_start = time.time()
#     
#     # ... simulation compute ...
#     
#     compute_time = time.time() - compute_start
#     frame_time = time.time() - frame_start
#     
#     benchmark.record_frame(frame_time, compute_time)
#
# stats = benchmark.generate_report()

if __name__ == "__main__":
    print("Performance benchmark module loaded. Import and use in simulation loop.")
