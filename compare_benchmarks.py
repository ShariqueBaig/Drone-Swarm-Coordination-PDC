"""
compare_benchmarks.py - Compare native vs containerized performance
Generates analysis and visualization of performance metrics
"""

import json
import csv
from pathlib import Path
from datetime import datetime
import sys

def load_benchmark_results(benchmarks_dir="benchmarks"):
    """Load all benchmark results from directory"""
    benchmarks_dir = Path(benchmarks_dir)
    results = {"native": [], "container": []}
    
    for json_file in benchmarks_dir.glob("*_benchmark_*.json"):
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
                if "native" in json_file.name:
                    results["native"].append((json_file.name, data))
                else:
                    results["container"].append((json_file.name, data))
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    
    return results

def analyze_performance(results):
    """Analyze and compare performance metrics"""
    if not results["native"] or not results["container"]:
        print("❌ Missing benchmark data. Run both native and container benchmarks first.")
        print("\nSteps:")
        print("1. python3 benchmark_native.py")
        print("2. docker-compose --profile benchmark up drone-swarm-benchmark")
        print("3. python3 compare_benchmarks.py")
        return
    
    # Get latest results
    native_file, native_data = results["native"][-1]
    container_file, container_data = results["container"][-1]
    
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON: NATIVE vs CONTAINERIZED")
    print("="*70)
    print(f"Native:       {native_file}")
    print(f"Container:    {container_file}")
    print("="*70)
    
    # Calculate differences
    metrics_to_compare = [
        ("fps_avg", "FPS (Average)"),
        ("fps_std", "FPS (Std Dev)"),
        ("frame_time_avg_ms", "Frame Time (avg, ms)"),
        ("frame_time_std_ms", "Frame Time (std, ms)"),
        ("cpu_avg", "CPU Usage (%)"),
        ("memory_avg_mb", "Memory (avg, MiB)"),
    ]
    
    print("\nMETRIC COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<30} {'Native':>15} {'Container':>15} {'Overhead':>8}")
    print("-" * 70)
    
    overhead_results = {}
    
    for metric_key, metric_name in metrics_to_compare:
        native_val = native_data.get(metric_key, 0)
        container_val = container_data.get(metric_key, 0)
        
        if native_val != 0:
            # For metrics where lower is better (time, usage %)
            if "avg" in metric_key or "std" in metric_key or "usage" in metric_name:
                if metric_key.startswith("fps"):
                    # For FPS, higher is better: calculate as (container/native - 1) * 100
                    overhead = ((container_val / native_val) - 1) * 100 if native_val != 0 else 0
                else:
                    # For time/usage, higher is worse: calculate as (container/native - 1) * 100
                    overhead = ((container_val / native_val) - 1) * 100 if native_val != 0 else 0
            else:
                overhead = 0
            
            overhead_results[metric_key] = overhead
            
            overhead_str = f"{overhead:+.1f}%"
            if overhead > 15:
                overhead_str += "  ⚠️"
            elif overhead > 5:
                overhead_str += "  ⚡"
            elif overhead < -5:
                overhead_str += "  ✅"
            
            print(f"{metric_name:<30} {native_val:>15.2f} {container_val:>15.2f} {overhead_str:>8}")
        else:
            print(f"{metric_name:<30} {'N/A':>15} {'N/A':>15} {'N/A':>8}")
    
    print("-" * 70)
    
    # Overall assessment
    print("\nOVERALL ASSESSMENT:")
    print("-" * 70)
    
    fps_overhead = overhead_results.get("fps_avg", 0)
    frame_time_overhead = overhead_results.get("frame_time_avg_ms", 0)
    cpu_overhead = overhead_results.get("cpu_avg", 0)
    memory_overhead = overhead_results.get("memory_avg_mb", 0)
    
    # Performance rating
    avg_overhead = abs(fps_overhead)  # Use FPS as primary metric
    
    if avg_overhead < 10:
        rating = "✅ EXCELLENT - Containerization has minimal impact"
        status = "PASS"
    elif avg_overhead < 15:
        rating = "🟡 GOOD - Containerization overhead is acceptable"
        status = "PASS"
    elif avg_overhead < 25:
        rating = "⚠️ ACCEPTABLE - Moderate performance degradation (acceptable for CPU-only)"
        status = "WARN"
    else:
        rating = "❌ POOR - Significant performance degradation"
        status = "FAIL"
    
    print(f"\nPerformance Rating: {rating}")
    print(f"Status: {status}")
    print(f"FPS Loss: {abs(fps_overhead):.1f}%")
    print(f"Frame Time Increase: {frame_time_overhead:.1f}%")
    print(f"CPU Increase: {cpu_overhead:.1f}%")
    print(f"Memory Increase: {memory_overhead:.1f}%")
    
    # Recommendations
    print("\nRECOMMENDATIONS (CPU-ONLY OPTIMIZATION):")
    print("-" * 70)
    
    recommendations = []
    
    if fps_overhead > 20:
        recommendations.append("• Increase CPU allocation in docker-compose.yml (cpus: 4)")
        recommendations.append("• Use CPU pinning: --cpuset-cpus=0-3")
        recommendations.append("• Close other applications consuming CPU")
        recommendations.append("• Check for thermal throttling")
    
    if cpu_overhead > 25:
        recommendations.append("• Increase CPU limit in docker-compose.yml (cpus: 4)")
        recommendations.append("• Consider reducing simulation complexity")
        recommendations.append("• Check for competing processes")
    
    if memory_overhead > 500:
        recommendations.append("• Increase memory limit in docker-compose.yml (mem_limit: 4g)")
        recommendations.append("• Check for memory leaks in simulation code")
    
    if not recommendations:
        recommendations.append("✅ Performance is optimal! No adjustments needed.")
    
    for rec in recommendations:
        print(rec)
    
    print("-" * 70)
    
    # Detailed metrics table
    print("\nDETAILED METRICS:")
    print("-" * 70)
    
    all_metrics = set(native_data.keys()) & set(container_data.keys())
    numeric_metrics = [k for k in all_metrics if isinstance(native_data.get(k), (int, float))]
    
    print(f"{'Metric':<40} {'Native':>15} {'Container':>15}")
    print("-" * 70)
    
    for metric in sorted(numeric_metrics):
        if metric not in ["duration_seconds", "total_frames", "stable_frames"]:
            try:
                native_val = native_data.get(metric, 0)
                container_val = container_data.get(metric, 0)
                if isinstance(native_val, (int, float)) and isinstance(container_val, (int, float)):
                    print(f"{metric:<40} {native_val:>15.2f} {container_val:>15.2f}")
            except (TypeError, ValueError):
                pass
    
    print("="*70 + "\n")
    
    return status, overhead_results

def generate_csv_report(results, output_file="comparison_report.csv"):
    """Generate CSV report of comparison"""
    if not results["native"] or not results["container"]:
        return
    
    native_file, native_data = results["native"][-1]
    container_file, container_data = results["container"][-1]
    
    with open(output_file, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Metric", "Native", "Container", "Overhead (%)", "Status"])
        
        metrics_to_compare = [
            ("fps_avg", "FPS (Average)"),
            ("fps_std", "FPS (Std Dev)"),
            ("frame_time_avg_ms", "Frame Time (avg, ms)"),
            ("frame_time_std_ms", "Frame Time (std, ms)"),
            ("cpu_avg", "CPU Usage (%)"),
            ("memory_avg_mb", "Memory (avg, MiB)"),
            ("gpu_avg", "GPU Usage (%)"),
        ]
        
        for metric_key, metric_name in metrics_to_compare:
            native_val = native_data.get(metric_key, 0)
            container_val = container_data.get(metric_key, 0)
            
            if native_val != 0:
                overhead = ((container_val / native_val) - 1) * 100
                status = "PASS" if abs(overhead) < 15 else "WARN" if abs(overhead) < 25 else "FAIL"
                writer.writerow([metric_name, f"{native_val:.2f}", f"{container_val:.2f}", 
                                f"{overhead:+.1f}", status])

if __name__ == "__main__":
    results = load_benchmark_results()
    status, overhead = analyze_performance(results)
    
    # Generate CSV report
    generate_csv_report(results)
    print("Comparison report saved to: comparison_report.csv\n")
    
    sys.exit(0 if status == "PASS" else 1)
