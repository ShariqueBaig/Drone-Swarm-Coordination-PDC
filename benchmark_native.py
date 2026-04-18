"""
benchmark_native.py - Baseline performance testing for native (non-containerized) simulation (CPU-Only)
Run this to establish baseline metrics before containerization for comparison
"""

import sys
import time
import numpy as np
import psutil
import os
from datetime import datetime
from pathlib import Path

class NativePerformanceBenchmark:
    def __init__(self, duration_seconds=60, output_dir="benchmarks"):
        self.duration_seconds = duration_seconds
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "start_time": datetime.now().isoformat(),
            "environment": "native",
            "fps_readings": [],
            "cpu_usage": [],
            "memory_usage": [],
            "frame_times": [],
        }
        self.frame_times = []
        self.start_time = None
        
    def get_system_metrics(self):
        """Capture current system performance metrics"""
        try:
            cpu = psutil.cpu_percent(interval=0.05)
            memory = psutil.virtual_memory()
            
            return {
                "cpu": cpu,
                "memory": memory.percent,
                "memory_mb": memory.used / (1024**2),
            }
        except Exception as e:
            print(f"Error collecting metrics: {e}")
            return None
    
    def record_frame(self, frame_time):
        """Record frame time"""
        self.frame_times.append(frame_time)
        
        if len(self.frame_times) % 30 == 0:
            # Log metrics every 30 frames
            fps = 1.0 / np.mean(self.frame_times[-30:]) if self.frame_times else 0
            self.metrics["fps_readings"].append(fps)
            self.metrics["frame_times"].extend(self.frame_times[-30:])
            
            metrics = self.get_system_metrics()
            if metrics:
                self.metrics["cpu_usage"].append(metrics["cpu"])
                self.metrics["memory_usage"].append(metrics["memory"])
    
    def print_progress(self, current_time):
        """Print benchmark progress"""
        elapsed = current_time - self.start_time
        progress = (elapsed / self.duration_seconds) * 100
        sys.stdout.write(f"\rBenchmarking... {progress:.1f}% complete ({elapsed:.1f}s / {self.duration_seconds}s)")
        sys.stdout.flush()
    
    def generate_report(self):
        """Generate comprehensive benchmark report"""
        if not self.frame_times:
            print("\nNo frame data collected")
            return {}
        
        self.metrics["end_time"] = datetime.now().isoformat()
        
        frame_times = np.array(self.frame_times)
        
        # Filter out startup/shutdown outliers (first and last 5% of frames)
        skip_frames = int(len(frame_times) * 0.05)
        stable_frames = frame_times[skip_frames:-skip_frames] if len(frame_times) > skip_frames * 2 else frame_times
        
        stats = {
            "duration_seconds": sum(frame_times),
            "total_frames": len(frame_times),
            "stable_frames": len(stable_frames),
            "fps_avg": 1.0 / np.mean(stable_frames) if len(stable_frames) > 0 else 0,
            "fps_min": 1.0 / np.max(stable_frames) if len(stable_frames) > 0 else 0,
            "fps_max": 1.0 / np.min(stable_frames) if len(stable_frames) > 0 else 0,
            "fps_std": np.std([1.0 / ft for ft in stable_frames]) if len(stable_frames) > 0 else 0,
            "frame_time_avg_ms": np.mean(stable_frames) * 1000,
            "frame_time_min_ms": np.min(stable_frames) * 1000,
            "frame_time_max_ms": np.max(stable_frames) * 1000,
            "frame_time_std_ms": np.std(stable_frames) * 1000,
            "frame_time_p95_ms": np.percentile(stable_frames, 95) * 1000,
            "frame_time_p99_ms": np.percentile(stable_frames, 99) * 1000,
            "cpu_avg": np.mean(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "cpu_peak": np.max(self.metrics["cpu_usage"]) if self.metrics["cpu_usage"] else 0,
            "memory_avg_mb": np.mean(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
            "memory_peak_mb": np.max(self.metrics["memory_usage"]) if self.metrics["memory_usage"] else 0,
        }
        
        # Save comprehensive metrics
        import json
        import csv
        
        metrics_file = self.output_dir / f"native_benchmark_{self.timestamp}.json"
        with open(metrics_file, "w") as f:
            json.dump({**self.metrics, **stats}, f, indent=2, default=str)
        
        csv_file = self.output_dir / f"native_benchmark_{self.timestamp}.csv"
        with open(csv_file, "w", newline='') as f:
            writer = csv.DictWriter(f, fieldnames=stats.keys())
            writer.writeheader()
            writer.writerow(stats)
        
        # Print summary
        print("\n\n" + "="*70)
        print("DRONE SWARM NATIVE PERFORMANCE BENCHMARK (CPU-ONLY)")
        print("="*70)
        print(f"Duration: {stats['duration_seconds']:.2f} seconds")
        print(f"Total Frames: {stats['total_frames']} (Stable: {stats['stable_frames']})")
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
        print(f"  P95:          {stats['frame_time_p95_ms']:.3f} ms")
        print(f"  P99:          {stats['frame_time_p99_ms']:.3f} ms")
        print(f"\nRESOURCE USAGE:")
        print(f"  CPU Average:  {stats['cpu_avg']:.1f}%")
        print(f"  CPU Peak:     {stats['cpu_peak']:.1f}%")
        print(f"  Memory Avg:   {stats['memory_avg_mb']:.1f} MiB")
        print(f"  Memory Peak:  {stats['memory_peak_mb']:.1f} MiB")
        print("="*70)
        print(f"Baseline metrics saved to: {metrics_file}")
        print("="*70 + "\n")
        
        return stats


def run_simple_benchmark(duration=60):
    """Run a simple performance test loop"""
    benchmark = NativePerformanceBenchmark(duration_seconds=duration)
    benchmark.start_time = time.time()
    
    print(f"Starting {duration}s native performance benchmark...")
    print("Note: This is a timing loop. For real simulation metrics, integrate with actual simulation.")
    
    frame_count = 0
    target_fps = 60
    target_frame_time = 1.0 / target_fps
    
    while time.time() - benchmark.start_time < duration:
        loop_start = time.time()
        
        # Simulate some work
        _ = np.random.random((1000, 1000))
        _ = np.dot(_, _.T)
        
        loop_time = time.time() - loop_start
        frame_count += 1
        
        benchmark.record_frame(loop_time)
        
        if frame_count % 30 == 0:
            benchmark.print_progress(time.time())
        
        # Sleep to target FPS (optional)
        if loop_time < target_frame_time:
            time.sleep(target_frame_time - loop_time)
    
    stats = benchmark.generate_report()
    return stats


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Native performance benchmark for drone swarm")
    parser.add_argument("--duration", type=int, default=60, help="Benchmark duration in seconds")
    parser.add_argument("--output", type=str, default="benchmarks", help="Output directory for results")
    
    args = parser.parse_args()
    
    try:
        stats = run_simple_benchmark(duration=args.duration)
        print("\nBenchmark complete! Results saved to 'benchmarks/' directory.")
        print("Compare with containerized results using compare_benchmarks.py")
    except KeyboardInterrupt:
        print("\nBenchmark interrupted by user")
        sys.exit(1)
