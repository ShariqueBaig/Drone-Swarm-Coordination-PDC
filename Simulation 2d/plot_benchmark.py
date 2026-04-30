"""
plot_neighbor_benchmark.py — Visualization of Neighbor-Finding Performance
Milestone 3 · Algorithm Comparison Graphs

Reads the CSV output from benchmark_neighbor_algos.py and generates
professional matplotlib graphs suitable for presentations.

Usage:
    python plot_neighbor_benchmark.py [csv_file]
    python plot_neighbor_benchmark.py neighbor_algo_benchmark.csv
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_benchmark_csv(filename):
    """Load benchmark results from CSV."""
    results = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'drone_count': int(row['drone_count']),
                'naive_mean_ms': float(row['naive_mean_ms']),
                'naive_std_ms': float(row['naive_std_ms']),
                'octree_mean_ms': float(row['octree_mean_ms']),
                'octree_std_ms': float(row['octree_std_ms']),
                'speedup': float(row['speedup']),
                'winner': row['winner']
            })
    return sorted(results, key=lambda x: x['drone_count'])


def plot_results(results, output_file="neighbor_algo_comparison.png"):
    """Generate professional comparison graphs."""
    
    # Extract data
    drone_counts = np.array([r['drone_count'] for r in results])
    naive_times = np.array([r['naive_mean_ms'] for r in results])
    naive_errs = np.array([r['naive_std_ms'] for r in results])
    octree_times = np.array([r['octree_mean_ms'] for r in results])
    octree_errs = np.array([r['octree_std_ms'] for r in results])
    speedups = np.array([r['speedup'] for r in results])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Color scheme
    color_naive = '#FF6B6B'  # Red
    color_octree = '#4ECDC4'  # Teal
    color_speedup = '#45B7D1'  # Blue
    
    # ─── SUBPLOT 1: Absolute Time Comparison ──────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.errorbar(drone_counts, naive_times, yerr=naive_errs, 
                 marker='o', label='Naive O(n²)', color=color_naive, linewidth=2.5, markersize=7)
    ax1.errorbar(drone_counts, octree_times, yerr=octree_errs, 
                 marker='s', label='KDTree/Octree', color=color_octree, linewidth=2.5, markersize=7)
    ax1.set_xlabel('Number of Drones', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Time per Update (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('Absolute Performance', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    # ─── SUBPLOT 2: Speedup Factor ────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [color_octree if s > 1.0 else color_naive for s in speedups]
    bars = ax2.bar(range(len(drone_counts)), speedups, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='No advantage')
    ax2.set_xlabel('Drone Count', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Speedup (Naive / Octree)', fontsize=11, fontweight='bold')
    ax2.set_title('Octree Advantage Factor', fontsize=12, fontweight='bold')
    ax2.set_xticks(range(len(drone_counts)))
    ax2.set_xticklabels([f'{c:,}' for c in drone_counts], rotation=45, ha='right')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, speedup) in enumerate(zip(bars, speedups)):
        label = f'{speedup:.2f}x'
        y_pos = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos + 0.05, label, 
                ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # ─── SUBPLOT 3: Time Growth (Linear Scale) ───────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(drone_counts, naive_times, marker='o', label='Naive (O(n²))', 
            color=color_naive, linewidth=2.5, markersize=7)
    ax3.plot(drone_counts, octree_times, marker='s', label='KDTree (O(n log n))', 
            color=color_octree, linewidth=2.5, markersize=7)
    ax3.fill_between(drone_counts, naive_times - naive_errs, naive_times + naive_errs, 
                     color=color_naive, alpha=0.2)
    ax3.fill_between(drone_counts, octree_times - octree_errs, octree_times + octree_errs, 
                     color=color_octree, alpha=0.2)
    ax3.set_xlabel('Number of Drones', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Time per Update (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('Linear Scale Comparison', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # ─── SUBPLOT 4: Complexity Analysis ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Fit complexity curves
    # Naive: O(n²)
    naive_fit = np.polyfit(np.log(drone_counts), np.log(naive_times), 2)
    naive_curve = np.exp(np.polyval(naive_fit, np.log(drone_counts)))
    
    # Octree: O(n log n)
    octree_fit = np.polyfit(np.log(drone_counts), np.log(octree_times), 1)
    octree_curve = np.exp(np.polyval(octree_fit, np.log(drone_counts)))
    
    ax4.loglog(drone_counts, naive_times, 'o', label='Naive O(n²)', color=color_naive, markersize=8)
    ax4.loglog(drone_counts, octree_times, 's', label='KDTree O(n log n)', color=color_octree, markersize=8)
    ax4.loglog(drone_counts, naive_curve, '--', color=color_naive, linewidth=2, alpha=0.7, label='Naive fit')
    ax4.loglog(drone_counts, octree_curve, '--', color=color_octree, linewidth=2, alpha=0.7, label='Octree fit')
    ax4.set_xlabel('Number of Drones (log scale)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Time per Update (ms, log scale)', fontsize=11, fontweight='bold')
    ax4.set_title('Algorithm Complexity (Log-Log)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    # Main title
    fig.suptitle('Neighbor-Finding Algorithm Performance Comparison\nNaive O(n²) vs KDTree/Octree', 
                fontsize=14, fontweight='bold', y=0.995)
    
    # Save
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n[PLOT] Saved to {output_file}")
    
    # Also show the plot
    plt.show()


def main():
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    else:
        csv_file = "neighbor_algo_benchmark.csv"
    
    try:
        results = load_benchmark_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: {csv_file} not found.")
        print("Run 'python benchmark_neighbor_algos.py' first to generate the benchmark data.")
        sys.exit(1)
    
    print(f"\nLoaded {len(results)} benchmark records from {csv_file}")
    plot_results(results)


if __name__ == "__main__":
    main()
