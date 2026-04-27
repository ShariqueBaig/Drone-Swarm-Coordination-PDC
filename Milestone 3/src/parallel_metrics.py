"""
parallel_metrics.py — Amdahl's Law & Gustafson's Law Analysis Module
PDC Project · Milestone 3

═══ PDC TECHNIQUE: Amdahl's Law & Gustafson's Law (Speedup/Scalability Analysis) ═══

This module instruments the simulation to measure:
  1. Serial fraction (s) — portion of update() that cannot be parallelized
  2. Parallel fraction (p = 1 - s) — portion that benefits from parallelism
  3. Amdahl's Speedup: S(N) = 1 / (s + p/N)  — fixed problem size
  4. Gustafson's Speedup: S(N) = s + p*N       — scaled problem size
"""

import time
import csv
import os
from collections import defaultdict

FLYNN_TAXONOMY = {
    "neighbor_find":       "SIMD",
    "compute_forces":      "SIMD",
    "obstacle_avoidance":  "SIMD",
    "boundary_avoidance":  "SIMD",
    "formation_steer":     "SIMD",
    "task_steer":          "SIMD",
    "auction":             "MIMD",
    "coverage_grid":       "SIMD",
    "collision_detect":    "SIMD",
    "integration":         "SIMD",
    "fork_join_dispatch":  "MIMD",
    "pipeline_overlap":    "MIMD",
    "gpu_transfer":        "SIMD",
    "serial_overhead":     "SISD",
}

class ParallelMetrics:
    def __init__(self, enabled=True, warmup_frames=50):
        self.enabled = enabled
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        self._section_times = defaultdict(list)
        self._section_starts = {}
        self._frame_serial_time = 0.0
        self._frame_parallel_time = 0.0
        self._frame_start = None
        self._serial_sections = {"serial_overhead", "auction"}
        self._parallel_sections = set(FLYNN_TAXONOMY.keys()) - self._serial_sections
        self._serial_fractions = []
        self._frame_total_times = []

    def start_frame(self):
        if not self.enabled: return
        self._frame_start = time.perf_counter()
        self._frame_serial_time = 0.0
        self._frame_parallel_time = 0.0

    def start_section(self, name):
        if not self.enabled: return
        self._section_starts[name] = time.perf_counter()

    def end_section(self, name):
        if not self.enabled or name not in self._section_starts: return
        elapsed = time.perf_counter() - self._section_starts.pop(name)
        if self.frame_count >= self.warmup_frames:
            self._section_times[name].append(elapsed)
            if name in self._serial_sections:
                self._frame_serial_time += elapsed
            else:
                self._frame_parallel_time += elapsed

    def end_frame(self):
        if not self.enabled or self._frame_start is None:
            self.frame_count += 1
            return
        total = time.perf_counter() - self._frame_start
        self.frame_count += 1
        if self.frame_count >= self.warmup_frames:
            frame_measured = self._frame_serial_time + self._frame_parallel_time
            if frame_measured > 1e-9:
                s = self._frame_serial_time / frame_measured
                self._serial_fractions.append(s)
            self._frame_total_times.append(total)

    def get_serial_fraction(self):
        if not self._serial_fractions: return 0.5
        return sum(self._serial_fractions) / len(self._serial_fractions)

    def amdahl_speedup(self, num_processors):
        s = self.get_serial_fraction()
        p = 1.0 - s
        return 1.0 / (s + p / num_processors)

    def gustafson_speedup(self, num_processors):
        s = self.get_serial_fraction()
        p = 1.0 - s
        return s + p * num_processors

    def get_section_breakdown(self):
        result = {}
        for name, times in self._section_times.items():
            if times: result[name] = (sum(times) / len(times)) * 1000
        return result

    def get_speedup_table(self, max_cores=16):
        rows = []
        for n in [1, 2, 4, 6, 8, 12, 16]:
            if n > max_cores: break
            rows.append((n, self.amdahl_speedup(n), self.gustafson_speedup(n)))
        return rows

    def get_hud_text(self):
        if not self.enabled or self.frame_count < self.warmup_frames + 10:
            return "Metrics: warming up..."
        s = self.get_serial_fraction()
        p = 1.0 - s
        a4 = self.amdahl_speedup(4)
        a8 = self.amdahl_speedup(8)
        g4 = self.gustafson_speedup(4)
        breakdown = self.get_section_breakdown()
        top3 = sorted(breakdown.items(), key=lambda x: -x[1])[:3]
        top3_str = "  ".join(f"{n}:{v:.1f}ms" for n, v in top3)
        return (f"Serial: {s:.1%}  Parallel: {p:.1%}\n"
                f"Amdahl(4): {a4:.2f}x  Amdahl(8): {a8:.2f}x\n"
                f"Gustafson(4): {g4:.2f}x\n"
                f"Hot: {top3_str}")

    def print_report(self):
        if not self._serial_fractions:
            print("[ParallelMetrics] No data collected yet.")
            return
        s = self.get_serial_fraction()
        p = 1.0 - s
        print("\n" + "=" * 70)
        print("  PDC PARALLEL PERFORMANCE ANALYSIS")
        print("  Amdahl's Law & Gustafson's Law Report")
        print("=" * 70)
        print(f"  Frames measured : {len(self._serial_fractions)}")
        print(f"  Serial fraction : {s:.4f} ({s:.1%})")
        print(f"  Parallel fraction: {p:.4f} ({p:.1%})")
        print()
        print("  +---------+---------------+-----------------+")
        print("  |  Cores  | Amdahl S(N)   | Gustafson S(N)  |")
        print("  +---------+---------------+-----------------+")
        for n, a_s, g_s in self.get_speedup_table():
            print(f"  |  {n:>5}  |  {a_s:>10.3f}x  |  {g_s:>12.3f}x  |")
        print("  +---------+---------------+-----------------+")
        print()
        print("  Section Timing Breakdown:")
        print("  " + "-" * 55)
        breakdown = self.get_section_breakdown()
        for name, ms in sorted(breakdown.items(), key=lambda x: -x[1]):
            flynn = FLYNN_TAXONOMY.get(name, "?")
            bar = "#" * int(min(ms * 2, 30))
            print(f"    {name:<25} {ms:>7.2f} ms  [{flynn}]  {bar}")
        print("=" * 70 + "\n")

    def export_csv(self, filename="parallel_analysis.csv"):
        breakdown = self.get_section_breakdown()
        s = self.get_serial_fraction()
        with open(filename, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["section", "avg_ms", "flynn_class", "serial_or_parallel"])
            for name, ms in sorted(breakdown.items(), key=lambda x: -x[1]):
                flynn = FLYNN_TAXONOMY.get(name, "?")
                sp = "serial" if name in self._serial_sections else "parallel"
                w.writerow([name, f"{ms:.4f}", flynn, sp])
            w.writerow([])
            w.writerow(["metric", "value"])
            w.writerow(["serial_fraction", f"{s:.6f}"])
            w.writerow(["parallel_fraction", f"{1-s:.6f}"])
            for n, a_s, g_s in self.get_speedup_table():
                w.writerow([f"amdahl_{n}_cores", f"{a_s:.4f}"])
                w.writerow([f"gustafson_{n}_cores", f"{g_s:.4f}"])
        print(f"[ParallelMetrics] Exported to {filename}")
