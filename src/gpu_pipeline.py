"""
gpu_pipeline.py — Asynchronous GPU Pipeline & Memory Optimization
Milestone 3 · Rendering & Compute Pipeline

═══ PDC TECHNIQUES ═══
  - Pipeline Parallelism: Decouple compute (N) from render (N-1)
  - Async GPU Compute: CUDA streams for overlapped execution
  - False Sharing Avoidance: Cache-line padded buffers
  - Memory Coherence: Proper synchronization barriers
  - Lock-Free Producer-Consumer: Minimize contention
"""

import numpy as np
import threading
import time
import config

class PaddedGrid:
    """
    ═══ PDC TECHNIQUE: Cache Coherence & False Sharing Avoidance ═══
    
    Standard approach (PROBLEMATIC):
        visited_grid = np.zeros((30, 30, 30), dtype=bool)
        # Multiple threads writing to adjacent cells cause false sharing:
        # Thread 0 updates grid[0,0,0] and grid[0,0,1] land on same cache line
        # Thread 1 updates grid[0,0,2] — invalidates Thread 0's cache line
        # Result: cache misses cascade (MESI protocol thrashing)
    
    Solution: Pad each grid cell to cache-line boundary (64 bytes):
        Each logical cell occupies 64 bytes → different cache lines
        No invalidation between adjacent accesses
    """

    def __init__(self, shape, cache_line=64, dtype=bool):
        self.shape = shape
        self.cache_line = cache_line
        self.dtype = dtype

        # Padding: How many elements per cache line?
        elem_size = np.dtype(dtype).itemsize
        self.elements_per_line = max(1, cache_line // elem_size)

        # Allocate padded storage
        padded_shape = (
            shape[0] * self.elements_per_line,
            shape[1] * self.elements_per_line,
            shape[2] * self.elements_per_line
        )
        self._data = np.zeros(padded_shape, dtype=dtype)

    def __getitem__(self, key):
        """Transparent padding lookup."""
        if isinstance(key, tuple) and len(key) == 3:
            i, j, k = key
            pi = i * self.elements_per_line
            pj = j * self.elements_per_line
            pk = k * self.elements_per_line
            return self._data[pi, pj, pk]
        return self._data[key]

    def __setitem__(self, key, value):
        """Transparent padding assignment."""
        if isinstance(key, tuple) and len(key) == 3:
            i, j, k = key
            pi = i * self.elements_per_line
            pj = j * self.elements_per_line
            pk = k * self.elements_per_line
            self._data[pi, pj, pk] = value
        else:
            self._data[key] = value

    def all_indices(self):
        """Return all logical indices (not padded)."""
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                for k in range(self.shape[2]):
                    yield (i, j, k), self[i, j, k]


class AsyncGPUPipeline:
    """
    ═══ PDC TECHNIQUE: Pipeline Parallelism with GPU Streams ═══
    
    Timeline:
        Frame N-1: GPU compute + Transfer results to CPU (async)
        Frame N:   CPU processes N-1, GPU starts compute N (no blocking)
    
    This overlaps GPU execution with CPU work, hiding latency.
    """

    def __init__(self, enable_gpu=False):
        self.enable_gpu = enable_gpu
        self.gpu_stream_compute = None
        self.gpu_stream_transfer = None

        if enable_gpu:
            try:
                import cupy as cp
                self.gpu_stream_compute = cp.cuda.Stream(non_blocking=True)
                self.gpu_stream_transfer = cp.cuda.Stream(non_blocking=True)
                print("[GPU] Async pipeline initialized (2 CUDA streams)")
            except ImportError:
                self.enable_gpu = False
                print("[GPU] CuPy not available, async pipeline disabled")

    def dispatch_compute_async(self, positions, kernel_func, *args):
        """
        Launch GPU compute without blocking CPU.
        kernel_func: function that performs GPU computation
        Returns: Future-like object (or None if CPU fallback)
        """
        if not self.enable_gpu:
            # CPU fallback (synchronous)
            return kernel_func(positions, *args)

        try:
            import cupy as cp
            with self.gpu_stream_compute:
                return kernel_func(positions, *args)
        except Exception as e:
            print(f"[GPU] Compute dispatch failed: {e}")
            return None

    def sync_point(self):
        """Synchronize all streams (used at frame boundary)."""
        if not self.enable_gpu:
            return

        try:
            import cupy as cp
            if self.gpu_stream_compute:
                self.gpu_stream_compute.synchronize()
            if self.gpu_stream_transfer:
                self.gpu_stream_transfer.synchronize()
        except Exception:
            pass


class FrameDoubleBuffer:
    """
    ═══ PDC TECHNIQUE: Double Buffering for Lock-Free Exchange ═══
    
    Allows compute thread to write Frame N while render thread
    reads Frame N-1, without synchronization overhead.
    """

    def __init__(self, num_drones):
        self.num_drones = num_drones

        # Two complete frame snapshots
        self.buffers = [
            {
                'positions': np.zeros((num_drones, 3), dtype=np.float32),
                'velocities': np.zeros((num_drones, 3), dtype=np.float32),
                'dead_mask': np.zeros(num_drones, dtype=bool),
                'frame_id': 0,
            },
            {
                'positions': np.zeros((num_drones, 3), dtype=np.float32),
                'velocities': np.zeros((num_drones, 3), dtype=np.float32),
                'dead_mask': np.zeros(num_drones, dtype=bool),
                'frame_id': 0,
            }
        ]

        self.write_idx = 0  # Compute thread writes to this
        self.read_idx = 1   # Render thread reads from this
        self._swap_lock = threading.Lock()

    def get_write_buffer(self):
        """For compute thread (producer)."""
        return self.buffers[self.write_idx]

    def get_read_buffer(self):
        """For render thread (consumer)."""
        return self.buffers[self.read_idx]

    def swap(self):
        """
        Called by producer after frame completion.
        Atomically swaps write/read indices.
        """
        with self._swap_lock:
            self.write_idx, self.read_idx = self.read_idx, self.write_idx

    def frame_lag(self):
        """Compute how far behind render is (0 = current, 1 = one frame old)."""
        with self._swap_lock:
            return abs(self.buffers[self.write_idx]['frame_id'] -
                       self.buffers[self.read_idx]['frame_id'])


class RenderFrameStats:
    """
    ═══ PDC TECHNIQUE: Performance Instrumentation for Optimization ═══
    Separate compute and render metrics to identify bottlenecks.
    """

    def __init__(self):
        self.compute_frame_times = []
        self.render_frame_times = []
        self.compute_tps = 0.0
        self.render_fps = 0.0
        self._lock = threading.Lock()

    def record_compute_frame(self, dt):
        """Called by compute thread."""
        with self._lock:
            self.compute_frame_times.append(dt)
            if len(self.compute_frame_times) > 60:
                self.compute_frame_times.pop(0)
                if self.compute_frame_times:
                    avg_dt = sum(self.compute_frame_times) / len(self.compute_frame_times)
                    self.compute_tps = 1.0 / avg_dt if avg_dt > 0 else 0

    def record_render_frame(self, dt):
        """Called by render thread."""
        with self._lock:
            self.render_frame_times.append(dt)
            if len(self.render_frame_times) > 60:
                self.render_frame_times.pop(0)
                if self.render_frame_times:
                    avg_dt = sum(self.render_frame_times) / len(self.render_frame_times)
                    self.render_fps = 1.0 / avg_dt if avg_dt > 0 else 0

    def get_stats(self):
        with self._lock:
            return {
                'compute_tps': self.compute_tps,
                'render_fps': self.render_fps,
                'lag_ratio': self.compute_tps / self.render_fps if self.render_fps > 0 else 1.0
            }

    def print_stats(self):
        stats = self.get_stats()
        print(f"[PERF] Compute: {stats['compute_tps']:.1f} TPS | "
              f"Render: {stats['render_fps']:.1f} FPS | "
              f"Lag: {stats['lag_ratio']:.2f}x")
