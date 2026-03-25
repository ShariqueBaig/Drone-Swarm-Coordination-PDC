"""
visualize_benchmark.py — Proper benchmark visualization
PDC Project · Spring 2026

WHAT WAS WRONG WITH THE ORIGINAL GRAPH:
  1. Only showed one method (no comparison) — FPS over time is meaningless
     without a baseline. It just showed warmup behavior, not optimization.
  2. The graph was from a DIFFERENT run than the CSV (old unoptimized data).
  3. No method annotations or comparison bars.
  4. Memory plot was nearly flat (54.9-55.1 MB range) — uninformative subplot.

THIS SCRIPT PRODUCES 4 PANELS:
  1. FPS over time (color per method) — shows stable performance per method
  2. Bar chart: mean ± std FPS per method — the headline comparison
  3. FPS distribution (boxplot) — shows variance, not just mean
  4. avg_neighbors vs FPS scatter — shows why FPS varies (swarm density)

USAGE:
    python visualize_benchmark.py                   # uses benchmark_comparison.csv
    python visualize_benchmark.py mydata.csv        # uses specified file
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

# ── Config ────────────────────────────────────────────────────────────────
CSV_FILE   = sys.argv[1] if len(sys.argv) > 1 else 'benchmark_comparison.csv'
OUT_FILE   = CSV_FILE.replace('.csv', '_plot.png')

METHOD_COLORS = {
    'naive':    '#E74C3C',   # red — slowest
    'grid':     '#3498DB',   # blue
    'quadtree': '#2ECC71',   # green — fastest
}
METHOD_ORDER = ['naive', 'grid', 'quadtree']


def load_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            f"Run benchmark_runner.py first to generate comparison data.")
    df = pd.read_csv(path)
    return df


def smooth(series, window=10):
    return series.rolling(window=window, center=True, min_periods=1).mean()


def make_figure(df):
    methods_present = [m for m in METHOD_ORDER if m in df['method'].unique()]
    n_methods       = len(methods_present)

    fig = plt.figure(figsize=(16, 10))
    fig.suptitle('Swarm Simulation — Method Performance Comparison\n'
                 f'N={df["num_drones"].iloc[0]} boids | '
                 f'Methods: {", ".join(methods_present)}',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.30,
                  top=0.91, bottom=0.08, left=0.07, right=0.96)

    ax_line = fig.add_subplot(gs[0, :])     # top full-width: FPS over frame
    ax_bar  = fig.add_subplot(gs[1, 0])     # bottom-left: bar chart
    ax_box  = fig.add_subplot(gs[1, 1])     # bottom-right: boxplot

    # ── Panel 1: FPS over frame per method ──────────────────────────────
    for method in methods_present:
        sub    = df[df['method'] == method].copy()
        color  = METHOD_COLORS.get(method, 'gray')
        frames = sub['frame'].values
        fps    = sub['fps'].values

        # Raw (faint) + smoothed (bold)
        ax_line.plot(frames, fps, color=color, alpha=0.18, linewidth=0.8)
        ax_line.plot(frames, smooth(sub['fps']), color=color,
                     linewidth=2.2, label=f'{method} (smoothed)')

    ax_line.set_title('FPS Over Time (raw + 10-frame rolling mean)',
                      fontsize=11, pad=8)
    ax_line.set_xlabel('Frame (after warmup)')
    ax_line.set_ylabel('FPS')
    ax_line.legend(loc='upper right', fontsize=9)
    ax_line.grid(True, alpha=0.25)
    ax_line.set_ylim(bottom=0)

    # Annotate method regions if data comes from a sequential single-run CSV
    if len(methods_present) == 1:
        ax_line.set_title(
            f'FPS Over Time — {methods_present[0]} only\n'
            '(run benchmark_runner.py for multi-method comparison)',
            fontsize=10)

    # ── Panel 2: Bar chart mean ± std ────────────────────────────────────
    means   = []
    stds    = []
    colors  = []
    labels  = []

    for method in methods_present:
        sub = df[df['method'] == method]['fps']
        means.append(sub.mean())
        stds.append(sub.std())
        colors.append(METHOD_COLORS.get(method, 'gray'))
        labels.append(method)

    x = np.arange(len(labels))
    bars = ax_bar.bar(x, means, yerr=stds, color=colors, alpha=0.82,
                      capsize=6, error_kw={'linewidth': 1.8},
                      edgecolor='white', linewidth=0.6)

    # Annotate bars with mean value and speedup vs naive
    naive_mean = means[labels.index('naive')] if 'naive' in labels else None
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        label_str = f'{mean:.0f} FPS'
        if naive_mean and labels[i] != 'naive' and naive_mean > 0:
            speedup = mean / naive_mean
            label_str += f'\n({speedup:.1f}×)'
        ax_bar.text(bar.get_x() + bar.get_width() / 2,
                    mean + std + max(means) * 0.02,
                    label_str, ha='center', va='bottom',
                    fontsize=9, fontweight='bold')

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, fontsize=10)
    ax_bar.set_title('Mean FPS ± Std Dev\n(higher is better)', fontsize=11)
    ax_bar.set_ylabel('FPS')
    ax_bar.set_ylim(bottom=0, top=max(means) * 1.3)
    ax_bar.grid(True, axis='y', alpha=0.25)
    ax_bar.set_axisbelow(True)

    # ── Panel 3: Boxplot ─────────────────────────────────────────────────
    data_by_method = [df[df['method'] == m]['fps'].values for m in methods_present]
    bp = ax_box.boxplot(data_by_method,
                        patch_artist=True,
                        medianprops=dict(color='white', linewidth=2.5),
                        whiskerprops=dict(linewidth=1.4),
                        capprops=dict(linewidth=1.4),
                        flierprops=dict(marker='o', markersize=3, alpha=0.4))

    for patch, method in zip(bp['boxes'], methods_present):
        patch.set_facecolor(METHOD_COLORS.get(method, 'gray'))
        patch.set_alpha(0.82)

    ax_box.set_xticklabels(methods_present, fontsize=10)
    ax_box.set_title('FPS Distribution\n(median, IQR, outliers)', fontsize=11)
    ax_box.set_ylabel('FPS')
    ax_box.grid(True, axis='y', alpha=0.25)
    ax_box.set_axisbelow(True)

    # ── Summary stats text ────────────────────────────────────────────────
    summary_lines = ['Method     Mean    Median   Min    Max']
    summary_lines.append('─' * 44)
    for method in methods_present:
        sub = df[df['method'] == method]['fps']
        summary_lines.append(
            f'{method:<10} {sub.mean():>6.1f}  {sub.median():>7.1f}  '
            f'{sub.min():>5.1f}  {sub.max():>6.1f}')

    if naive_mean and len(methods_present) > 1:
        summary_lines.append('─' * 44)
        for method in methods_present:
            if method == 'naive': continue
            speedup = df[df['method']==method]['fps'].mean() / naive_mean
            summary_lines.append(f'{method} speedup vs naive: {speedup:.2f}×')

    fig.text(0.5, 0.01,
             '  |  '.join(summary_lines[:2]) + '\n' +
             '  |  '.join(summary_lines[2:]),
             ha='center', va='bottom', fontsize=7.5,
             fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.4', facecolor='#f8f8f8',
                       edgecolor='#cccccc', alpha=0.9))

    return fig


def main():
    print(f"Loading: {CSV_FILE}")
    df  = load_data(CSV_FILE)
    print(f"Rows: {len(df)} | Methods: {list(df['method'].unique())}")
    print(f"FPS range: {df['fps'].min():.1f} – {df['fps'].max():.1f}")

    # If this is a single-run log (not the benchmark runner output),
    # print a warning and still plot what we have
    methods = df['method'].unique()
    if len(methods) == 1:
        print(f"\n⚠  Only one method found ({methods[0]}).")
        print("   The graph will show FPS over time but cannot compare methods.")
        print("   Run benchmark_runner.py for a proper multi-method comparison.\n")

    fig = make_figure(df)
    fig.savefig(OUT_FILE, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved: {OUT_FILE}")
    plt.close(fig)


if __name__ == '__main__':
    main()
