"""
spatial_utils.py — HPC Spatial Optimizations
PDC Project · Milestone 3

This module implements two advanced HPC techniques:
1. Morton Encoding (Z-order curve) for Cache Locality.
2. Numba JIT Compilation for CPU fast-paths (Loop Fusion).

═══ PDC TECHNIQUE: Cache Locality (Morton Space-Filling Curves) ═══
Memory layout dictates performance. By interleaving the discrete bits
of drones' X, Y, Z coordinates, we generate a Morton Code. Sorting the
drones by this code ensures that drones which are physically close in 3D
space are stored contiguously in memory. This maximizes CPU L1/L2 cache
hit rates during O(N) or O(N^2) neighbor discovery.

═══ PDC TECHNIQUE: JIT Compilation & Loop Fusion (Numba) ═══
Instead of allocating multiple intermediate arrays in memory (which NumPy
does for complex vectorized formulas like `a = b * c + d`), Numba compiles
the loop to LLVM machine code. The CPU registers are reused, drastically
reducing memory bandwidth pressure.
"""

import numpy as np
import math

# ── Numba JIT (Optional Backend) ─────────────────────────────────────────────
NUMBA_AVAILABLE = False
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
    print("[HPC] Numba JIT acceleration enabled.")
except ImportError:
    print("[HPC] Numba not found. Using NumPy CPU fallback.")
    # Dummy decorator if numba is missing
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    prange = range

# ═════════════════════════════════════════════════════════════════════════════
#  MORTON ENCODING (CACHE LOCALITY)
# ═════════════════════════════════════════════════════════════════════════════

@njit(cache=False)
def expand_bits(v):
    """
    Spreads bits of a 10-bit integer across 30 bits.
    e.g., v = 00000000000000000000001111111111 ->
          001001001001001001001001001001001
    """
    v = (v | (v << 16)) & 0x030000FF
    v = (v | (v <<  8)) & 0x0300F00F
    v = (v | (v <<  4)) & 0x030C30C3
    v = (v | (v <<  2)) & 0x09249249
    return v

@njit(cache=False)
def compute_morton_codes_3d(positions, extent_min, extent_max):
    """
    Computes a 30-bit Morton code for each 3D position to ensure cache locality.
    positions: (N, 3) float array
    extent_min: minimum coordinate of the bounding box
    extent_max: maximum coordinate of the bounding box
    Returns: (N,) array of uint32 Morton codes
    """
    n = positions.shape[0]
    codes = np.zeros(n, dtype=np.uint32)
    
    scale = 1023.0 / (extent_max - extent_min)
    
    for i in range(n):
        # Normalize coordinates to 0..1023 (10 bits per axis)
        x = max(0.0, min(1023.0, (positions[i, 0] - extent_min) * scale))
        y = max(0.0, min(1023.0, (positions[i, 1] - extent_min) * scale))
        z = max(0.0, min(1023.0, (positions[i, 2] - extent_min) * scale))
        
        ix = int(x)
        iy = int(y)
        iz = int(z)
        
        xx = expand_bits(ix)
        yy = expand_bits(iy)
        zz = expand_bits(iz)
        
        # Interleave bits: ZZ YY XX
        codes[i] = (zz << 2) | (yy << 1) | xx
        
    return codes

def sort_by_morton(positions, velocities, mission_type, assigned_tasks, ext_min=0.0, ext_max=1000.0):
    """
    Sorts swarm state arrays in-place to enforce spatial cache locality.
    ═══ PDC TECHNIQUE: Architecture Tuning (Cache Locality) ═══
    Args:
        positions: (N, 3) array
        velocities: (N, 3) array
        mission_type: (N,) array
        assigned_tasks: (N,) array
    Returns:
        sort_indices (to update mappings if needed)
    """
    # 1. Compute codes
    codes = compute_morton_codes_3d(positions, ext_min, ext_max)
    
    # 2. Find sort order
    sort_idx = np.argsort(codes)
    
    # 3. Apply sorted order (reordering memory physically)
    # Using np.copy ensures elements are placed contiguously based on the new order
    positions[:] = positions[sort_idx]
    velocities[:] = velocities[sort_idx]
    mission_type[:] = mission_type[sort_idx]
    assigned_tasks[:] = assigned_tasks[sort_idx]
    
    return sort_idx

# ═════════════════════════════════════════════════════════════════════════════
#  NUMBA JIT FAST PATHS (LOOP FUSION)
# ═════════════════════════════════════════════════════════════════════════════

@njit(parallel=True, fastmath=True, cache=False)
def fast_distance_matrix(pos, r_sq):
    """
    Computes pairwise squared distances using OpenMP-style nested loops.
    Returns (ii, jj) pair arrays.
    
    ═══ PDC TECHNIQUE: JIT Compilation & Loop Fusion ═══
    This completely avoids the massive (N, N, 3) intermediate array created
    by NumPy broadcasting `pos[:, None] - pos[None, :]`.
    """
    n = pos.shape[0]
    # We don't know output size, so we preallocate max possible edges (N^2/2)
    # and truncate. For large N, we might want dynamic sizing in a real app,
    # but for N=100 it's trivial (10000 ints).
    max_pairs = (n * (n - 1)) // 2
    ii = np.zeros(max_pairs, dtype=np.int32)
    jj = np.zeros(max_pairs, dtype=np.int32)
    
    count = 0
    # prange distributes the outer loop across available CPU cores (MIMD)
    # Note: adding pairs to shared array requires careful thread safety in Numba.
    # To keep it completely race-free within Numba, we do serial loop here,
    # or chunk carefully. A serial Numba loop for N=100 is < 50 microseconds.
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dz = pos[i, 2] - pos[j, 2]
            d2 = dx*dx + dy*dy + dz*dz
            if d2 < r_sq:
                ii[count] = i
                jj[count] = j
                count += 1
                
    return ii[:count], jj[:count]
