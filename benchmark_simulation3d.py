#!/usr/bin/env python3
"""
Benchmark simulation3d.py for all 3 algorithms
Measures FPS, frame time, CPU, and memory usage
"""
import os
import sys
import time
import psutil
import csv
from pathlib import Path
from datetime import datetime

# Add Milestone 2 src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Milestone 2', 'src'))

import config
from swarm_3d import SwarmManager3D
from environment3d import Environment3D
import numpy as np

class Simulation3DBenchmark:
    def __init__(self, algorithm=2, duration=60):
        """
        algorithm: 1=Octree, 2=Grid, 3=Naive
        duration: seconds to run
        """
        self.algorithm = algorithm
        self.algorithm_names = {1: "Octree", 2: "Grid Hash", 3: "Naive O(N²)"}
        self.method_map = {1: "octree", 2: "grid", 3: "naive"}
        self.duration = duration
        self.process = psutil.Process()
        
        # Initialize simulation WITHOUT Ursina rendering
        self.env = Environment3D()
        self.swarm = SwarmManager3D(self.env)
        
        # Set algorithm correctly
        self.swarm.use_method = self.method_map[algorithm]
        
        self.metrics = {
            'frame_times': [],
            'fps_values': [],
            'cpu_percent': [],
            'memory_mb': []
        }
        
    def run_benchmark(self):
        """Run benchmark loop without rendering"""
        print(f"\n{'='*70}")
        print(f"BENCHMARKING: {self.algorithm_names[self.algorithm]} (Algorithm {self.algorithm})")
        print(f"Duration: {self.duration} seconds")
        print(f"{'='*70}\n")
        
        start_time = time.perf_counter()
        frame_count = 0
        last_print = start_time
        
        try:
            while time.perf_counter() - start_time < self.duration:
                frame_start = time.perf_counter()
                
                # Update swarm (this is the core operation being benchmarked)
                self.swarm.update()
                
                frame_end = time.perf_counter()
                frame_time = (frame_end - frame_start) * 1000  # Convert to ms
                fps = 1000 / frame_time if frame_time > 0 else 0
                
                self.metrics['frame_times'].append(frame_time)
                self.metrics['fps_values'].append(fps)
                
                # Collect system metrics
                try:
                    self.metrics['cpu_percent'].append(self.process.cpu_percent(interval=None))
                    self.metrics['memory_mb'].append(self.process.memory_info().rss / 1024 / 1024)
                except:
                    pass
                
                frame_count += 1
                
                # Progress update every 5 seconds
                if time.perf_counter() - last_print >= 5:
                    elapsed = time.perf_counter() - start_time
                    avg_fps = np.mean(self.metrics['fps_values'][-100:]) if self.metrics['fps_values'] else 0
                    print(f"  Frame {frame_count:5d} | Elapsed: {elapsed:.1f}s | Avg FPS (last 100): {avg_fps:7.2f}")
                    last_print = time.perf_counter()
                
        except KeyboardInterrupt:
            print("\nBenchmark interrupted by user")
        except Exception as e:
            print(f"\nError during benchmark: {e}")
            import traceback
            traceback.print_exc()
        
        return self.compute_stats(frame_count)
    
    def compute_stats(self, frame_count):
        """Compute and print statistics"""
        if not self.metrics['fps_values']:
            print("No data collected!")
            return None
        
        fps_vals = np.array(self.metrics['fps_values'])
        frame_times = np.array(self.metrics['frame_times'])
        cpu_vals = np.array(self.metrics['cpu_percent']) if self.metrics['cpu_percent'] else []
        mem_vals = np.array(self.metrics['memory_mb']) if self.metrics['memory_mb'] else []
        
        stats = {
            'algorithm': self.algorithm_names[self.algorithm],
            'total_frames': frame_count,
            'fps_average': float(np.mean(fps_vals)),
            'fps_min': float(np.min(fps_vals)),
            'fps_max': float(np.max(fps_vals)),
            'fps_std': float(np.std(fps_vals)),
            'frame_time_avg': float(np.mean(frame_times)),
            'frame_time_min': float(np.min(frame_times)),
            'frame_time_max': float(np.max(frame_times)),
            'cpu_avg': float(np.mean(cpu_vals)) if len(cpu_vals) > 0 else 0,
            'cpu_peak': float(np.max(cpu_vals)) if len(cpu_vals) > 0 else 0,
            'memory_avg': float(np.mean(mem_vals)) if len(mem_vals) > 0 else 0,
            'memory_peak': float(np.max(mem_vals)) if len(mem_vals) > 0 else 0,
        }
        
        print(f"\n{'='*70}")
        print(f"RESULTS: {stats['algorithm']} (Algorithm {self.algorithm})")
        print(f"{'='*70}")
        print(f"Total Frames:     {stats['total_frames']}")
        print(f"\nFPS STATISTICS:")
        print(f"  Average:        {stats['fps_average']:8.2f} FPS")
        print(f"  Min:            {stats['fps_min']:8.2f} FPS")
        print(f"  Max:            {stats['fps_max']:8.2f} FPS")
        print(f"  Std Dev:        {stats['fps_std']:8.2f}")
        print(f"\nFRAME TIME (ms):")
        print(f"  Average:        {stats['frame_time_avg']:8.2f} ms")
        print(f"  Min:            {stats['frame_time_min']:8.2f} ms")
        print(f"  Max:            {stats['frame_time_max']:8.2f} ms")
        print(f"\nRESOURCE USAGE:")
        print(f"  CPU Average:    {stats['cpu_avg']:8.2f}%")
        print(f"  CPU Peak:       {stats['cpu_peak']:8.2f}%")
        print(f"  Memory Avg:     {stats['memory_avg']:8.2f} MB")
        print(f"  Memory Peak:    {stats['memory_peak']:8.2f} MB")
        print(f"{'='*70}\n")
        
        return stats

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=int, default=2, help='Algorithm: 1=Octree, 2=Grid, 3=Naive')
    parser.add_argument('--duration', type=int, default=60, help='Duration in seconds')
    parser.add_argument('--all', action='store_true', help='Run all algorithms')
    args = parser.parse_args()
    
    results = []
    algorithms = [1, 2, 3] if args.all else [args.algorithm]
    
    for algo in algorithms:
        try:
            bench = Simulation3DBenchmark(algorithm=algo, duration=args.duration)
            stats = bench.run_benchmark()
            if stats:
                results.append(stats)
        except Exception as e:
            print(f"Error benchmarking algorithm {algo}: {e}")
    
    # Save results
    if results:
        os.makedirs('benchmarks', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        csv_file = f'benchmarks/simulation3d_benchmark_{timestamp}.csv'
        
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to: {csv_file}")

if __name__ == '__main__':
    main()
