"""
IC(0) preconditioner: Python wrapper around ``ic0.c``.

The point of IC(0) in this study is that its sparsity pattern is *fixed* to
that of the lower triangle of A. Unlike a threshold ILU, it cannot pay for a
bad numbering with extra fill: every numbering gets exactly the same number
of non-zeros, and the only thing that can differ is the quality of the
preconditioner, which shows up in the iteration count. That is what isolates
the effect of the ordering.

Breakdown (a non-positive pivot) is handled by the standard Manteuffel
shift: factorise A + alpha*diag(A) with alpha increased until the
factorisation succeeds. The shift actually used is reported, because it is
itself ordering-dependent and it would be dishonest to hide it.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

__all__ = ["IC0", "ic0_factor"]

_LIB = None


def _lib():
    global _LIB
    if _LIB is None:
        path = os.path.join(os.path.dirname(__file__), "libic0.so")
        if not os.path.exists(path):
            raise RuntimeError(
                "libic0.so is missing; build it with\n"
                "  cc -O2 -shared -fPIC -o libic0.so ic0.c -lm")
        lib = ctypes.CDLL(path)
        lib.ic0_factor.argtypes = [
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double)]
        lib.ic0_factor.restype = ctypes.c_longlong
        lib.ic0_solve.argtypes = [
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double)]
        lib.ic0_solve.restype = None
        _LIB = lib
    return _LIB


class IC0:
    """An IC(0) factorisation usable as a SciPy ``LinearOperator``."""

    def __init__(self, Lp, Li, Lx, shift, attempts):
        self.Lp, self.Li, self.Lx = Lp, Li, Lx
        self.shift = shift
        self.attempts = attempts
        self.n = Lp.size - 1
        self.nnz = int(Lx.size)
        self._p = Lp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._i = Li.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))
        self._x = Lx.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    def solve(self, b):
        b = np.ascontiguousarray(b, dtype=np.float64)
        x = np.empty_like(b)
        _lib().ic0_solve(
            ctypes.c_longlong(self.n), self._p, self._i, self._x,
            b.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            x.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        return x

    def as_operator(self):
        return spla.LinearOperator((self.n, self.n), self.solve)


def ic0_factor(A, shift0=0.0, max_attempts=12):
    """IC(0) of a symmetric positive definite CSR matrix.

    Returns an :class:`IC0`. The pattern is that of ``tril(A)``, so
    ``result.nnz`` is the same for every ordering of the same matrix.
    """
    A = sp.csr_matrix(A)
    A.sum_duplicates()
    L0 = sp.tril(A, k=0, format="csr")
    L0.sort_indices()
    Lp = np.ascontiguousarray(L0.indptr, dtype=np.int32)
    Li = np.ascontiguousarray(L0.indices, dtype=np.int32)
    base = np.ascontiguousarray(L0.data, dtype=np.float64)

    diag_pos = Lp[1:] - 1                    # diagonal is last in each row
    diag = base[diag_pos].copy()

    shift = shift0
    for attempt in range(max_attempts):
        Lx = base.copy()
        if shift > 0.0:
            Lx[diag_pos] = diag * (1.0 + shift)
        info = _lib().ic0_factor(
            ctypes.c_longlong(L0.shape[0]),
            Lp.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            Li.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            Lx.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
        if info == 0:
            return IC0(Lp, Li, Lx, shift, attempt + 1)
        shift = 1e-3 if shift == 0.0 else 2.0 * shift
    raise RuntimeError(f"IC(0) broke down at every shift up to {shift}")
