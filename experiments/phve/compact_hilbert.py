"""
The compact Hilbert index of Hamilton & Rau-Chaplin (2008), implemented
here so that the claim made elsewhere in this repository -- that our
restricted-cube anisotropic order coincides with theirs -- can be tested
instead of asserted.

Reference
---------
C. H. Hamilton and A. Rau-Chaplin, "Compact Hilbert indices: space-filling
curves for domains with unequal side lengths", Inf. Process. Lett. 105
(2008) 155-163, and the companion technical report *Compact Hilbert indices
for multi-dimensional data* (Dalhousie CS-2006-07), Algorithms 1 and 2.

Everything below is a direct transcription of those algorithms.  No
novelty is claimed for any of it.  It is scalar and slow on purpose: it is
a reference implementation used to check a faster one, so clarity beats
speed.

Conventions
-----------
* ``n`` is the dimension; bit ``j`` of a level word corresponds to axis
  ``j``.
* ``m[j]`` is the number of bits of axis ``j``; ``M = max_j m[j]``.
* The standard Hilbert index in this family is exactly the compact index
  with ``m[j] = M`` for every ``j``.  Comparing against *that* -- rather
  than against our Skilling kernel -- isolates the question of compactness
  from the question of which of the many Hilbert variants is being used.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gray_code", "gray_code_inverse", "entry_point", "direction",
    "hilbert_index_hrc", "compact_hilbert_index", "compact_hilbert_order",
]


# ----------------------------------------------------------------------
# Bit utilities
# ----------------------------------------------------------------------

def _mask(n: int) -> int:
    return (1 << n) - 1


def rotate_right(b: int, i: int, n: int) -> int:
    i %= n
    return ((b >> i) | (b << (n - i))) & _mask(n)


def rotate_left(b: int, i: int, n: int) -> int:
    i %= n
    return ((b << i) | (b >> (n - i))) & _mask(n)


def gray_code(i: int) -> int:
    return i ^ (i >> 1)


def gray_code_inverse(g: int, n: int) -> int:
    """Inverse binary-reflected Gray code on n bits."""
    i = g
    s = 1
    while s < n:
        i ^= (i >> s)
        s <<= 1
    return i & _mask(n)


def _trailing_ones(i: int) -> int:
    """g(i): the index of the bit that flips between gc(i) and gc(i+1),
    i.e. the number of trailing 1 bits of i."""
    c = 0
    while (i >> c) & 1:
        c += 1
    return c


def entry_point(i: int) -> int:
    """e(i): e(0) = 0, e(i) = gc(2*floor((i-1)/2)) otherwise."""
    if i == 0:
        return 0
    return gray_code(2 * ((i - 1) // 2))


def direction(i: int, n: int) -> int:
    """d(i): 0 for i = 0; g(i-1) mod n for i even; g(i) mod n for i odd."""
    if i == 0:
        return 0
    if i % 2 == 0:
        return _trailing_ones(i - 1) % n
    return _trailing_ones(i) % n


def _T(e: int, d: int, b: int, n: int) -> int:
    return rotate_right(b ^ e, d + 1, n)


def _extract_mask(m, i: int, n: int) -> int:
    """Bit j set iff axis j still has a bit at level i, i.e. m[j] > i."""
    mu = 0
    for j in range(n):
        if m[j] > i:
            mu |= (1 << j)
    return mu


def _gray_code_rank(mu: int, w: int, n: int) -> int:
    """Compress the bits of w selected by mu, most significant axis first."""
    r = 0
    for j in range(n - 1, -1, -1):
        if (mu >> j) & 1:
            r = (r << 1) | ((w >> j) & 1)
    return r


# ----------------------------------------------------------------------
# The two indices
# ----------------------------------------------------------------------

def hilbert_index_hrc(p, n: int, M: int) -> int:
    """Standard Hilbert index, Hamilton & Rau-Chaplin Algorithm 1.

    ``p`` is a sequence of n non-negative integers, each < 2^M.
    """
    h = 0
    e = 0
    d = 0
    for i in range(M - 1, -1, -1):
        l = 0
        for j in range(n):
            l |= ((int(p[j]) >> i) & 1) << j
        l = _T(e, d, l, n)
        w = gray_code_inverse(l, n)
        e = e ^ rotate_left(entry_point(w), d + 1, n)
        d = (d + direction(w, n) + 1) % n
        h = (h << n) | w
    return h


def compact_hilbert_index(p, m, n: int) -> int:
    """Compact Hilbert index, Hamilton & Rau-Chaplin Algorithm 2.

    ``p[j]`` must satisfy 0 <= p[j] < 2^{m[j]}.  The result ranges over
    {0, ..., 2^{sum_j m[j]} - 1}.
    """
    M = max(m)
    h = 0
    e = 0
    d = 0
    for i in range(M - 1, -1, -1):
        mu = _extract_mask(m, i, n)
        mu_r = rotate_right(mu, d + 1, n)
        width = bin(mu_r).count("1")
        l = 0
        for j in range(n):
            l |= ((int(p[j]) >> i) & 1) << j
        l = _T(e, d, l, n)
        w = gray_code_inverse(l, n)
        r = _gray_code_rank(mu_r, w, n)
        e = e ^ rotate_left(entry_point(w), d + 1, n)
        d = (d + direction(w, n) + 1) % n
        h = (h << width) | r
    return h


def compact_hilbert_order(points_grid: np.ndarray, m) -> np.ndarray:
    """Permutation sorting integer grid points by compact Hilbert index."""
    n = points_grid.shape[1]
    idx = np.array([compact_hilbert_index(row, m, n)
                    for row in points_grid], dtype=object)
    return np.argsort(idx, kind="stable"), idx


def hrc_cube_order(points_grid: np.ndarray, M: int) -> np.ndarray:
    """Permutation sorting integer grid points by the *standard* H&RC index
    on the enclosing cube of order M -- the 'restricted-cube' construction,
    expressed in H&RC's own Hilbert variant."""
    n = points_grid.shape[1]
    idx = np.array([hilbert_index_hrc(row, n, M) for row in points_grid],
                   dtype=object)
    return np.argsort(idx, kind="stable"), idx
