"""
Orderings and per-time-step measurements for the comparative protocol.

Orderings compared
------------------
natural : the insertion order of the mesh vertices (no work at all).
rcm     : reverse Cuthill-McKee, *including* the construction of the
          adjacency graph from the element connectivity, which is the
          honest accounting: RCM cannot be run without that graph.
phve    : the PHVE / Hilbert order, which needs only the coordinates.

Measurements
------------
bandwidth, profile, ILU fill, PCG iteration count, simulated cache misses,
wall-clock times (renumbering, assembly, factorisation, solve).
"""

from __future__ import annotations

import ctypes
import os
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import reverse_cuthill_mckee

from .hilbert import phve_order, phve_order_aniso

__all__ = [
    "order_natural", "order_rcm", "order_phve", "order_phve_aniso",
    "apply_permutation", "bandwidth", "profile", "ilu_fill",
    "simulate_cache_misses", "Timer",
]

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = os.path.join(os.path.dirname(__file__), "cachesim.so")
        if not os.path.exists(path):
            raise RuntimeError(
                "cachesim.so is missing; build it with\n"
                "  cc -O2 -shared -fPIC -o cachesim.so cachesim.c")
        lib = ctypes.CDLL(path)
        lib.simulate_spmv.argtypes = [
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
            ctypes.c_longlong, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
        ]
        lib.simulate_spmv.restype = None
        _LIB = lib
    return _LIB


class Timer:
    """Context manager recording the elapsed wall-clock time in seconds."""

    def __init__(self):
        self.t = None

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.t = time.perf_counter() - self._t0
        return False


# ----------------------------------------------------------------------
# Orderings
# ----------------------------------------------------------------------

def order_natural(mesh):
    """Identity permutation, timed for completeness (cost = 0)."""
    with Timer() as tm:
        perm = np.arange(mesh.n_vertices)
    return perm, tm.t


def _adjacency_from_tets(tets, n):
    from .mesh import unique_edges
    pairs = unique_edges(tets, n)
    data = np.ones(pairs.shape[0] * 2, dtype=np.int8)
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    return sp.csr_matrix((data, (rows, cols)), shape=(n, n))


def order_rcm(mesh, graph=None):
    """Reverse Cuthill-McKee.  The timing includes building the graph
    unless a pre-built one is supplied (which is the situation when the
    assembled matrix pattern happens to be available already)."""
    with Timer() as tm:
        G = _adjacency_from_tets(mesh.tets, mesh.n_vertices) if graph is None else graph
        order = reverse_cuthill_mckee(G, symmetric_mode=True)
        perm = np.asarray(order)
    return perm, tm.t


def order_phve(mesh, p):
    with Timer() as tm:
        perm = phve_order(mesh.points, mesh.box, p)
    return perm, tm.t


def order_phve_aniso(mesh, orders):
    with Timer() as tm:
        perm = phve_order_aniso(mesh.points, mesh.box, orders)
    return perm, tm.t


def apply_permutation(A, perm):
    """Return P A P^T where P is the permutation matrix sending the old
    index perm[k] to the new index k."""
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    A = A.tocoo()
    B = sp.coo_matrix((A.data, (inv[A.row], inv[A.col])), shape=A.shape)
    return B.tocsr()


# ----------------------------------------------------------------------
# Sparsity measurements
# ----------------------------------------------------------------------

def bandwidth(A) -> int:
    A = A.tocoo()
    if A.nnz == 0:
        return 0
    return int(np.abs(A.row.astype(np.int64) - A.col.astype(np.int64)).max())


def profile(A) -> int:
    """sum_i (i - min{j : A_ij != 0}); the storage of a skyline solver."""
    A = A.tocsr()
    n = A.shape[0]
    out = 0
    indptr, indices = A.indptr, A.indices
    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if e > s:
            out += i - int(indices[s:e].min())
    return int(out)


def ilu_fill(A, drop_tol=1e-4, fill_factor=10.0):
    """nnz of an incomplete LU factorisation, and the factor object."""
    ilu = spla.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    return int(ilu.L.nnz + ilu.U.nnz), ilu


def simulate_cache_misses(A, line=64, elem=8, sets=64, ways=8):
    """Simulated LRU cache misses of one CSR SpMV with this ordering.

    Defaults model a 32 KiB, 8-way, 64-byte-line L1 data cache
    (64 sets * 8 ways * 64 B = 32 KiB), matching the machine used.
    """
    A = A.tocsr()
    indptr = np.ascontiguousarray(A.indptr, dtype=np.int32)
    indices = np.ascontiguousarray(A.indices, dtype=np.int32)
    out = (ctypes.c_longlong * 2)()
    _lib().simulate_spmv(
        indptr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        ctypes.c_longlong(A.shape[0]), ctypes.c_longlong(elem),
        ctypes.c_longlong(line), ctypes.c_longlong(sets),
        ctypes.c_longlong(ways), out)
    return {"accesses": int(out[0]), "misses": int(out[1]),
            "miss_rate": out[1] / max(out[0], 1)}
