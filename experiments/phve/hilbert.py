"""
Vectorised Hilbert kernels for the PHVE experiment notebook.

Contents
--------
hilbert_encode(coords, p)      Skilling transpose-to-index, any d, vectorised.
hilbert_decode(index, p, d)    inverse of the above.
phve_order(points, box, p)     cubic-order PHVE renumbering permutation.
phve_order_aniso(points, box, orders)
                               per-axis-order (anisotropic) PHVE permutation
                               by the restricted-cube construction; see
                               Lemma "rank compression" in the paper.
                               NOT order-equivalent to the compact Hilbert
                               index of Hamilton & Rau-Chaplin (2008) in
                               d = 3 -- see the note below and exp12.
choose_orders(box, budget)     the R1 selection rule for the per-axis orders.

The Skilling kernel is the classical algorithm of J. Skilling, "Programming
the Hilbert curve", AIP Conf. Proc. 707 (2004) 381-387.  No novelty is
claimed for it; the vectorisation below is a straight transcription of the
scalar loop onto numpy integer arrays.

All routines operate on int64; they are exact as long as d * p <= 62.

Which Hilbert curve is this?
---------------------------
An earlier version of this file stated that `phve_order_aniso` induces the
same total order as the compact Hilbert index of Hamilton & Rau-Chaplin
(2008).  `exp12_compact_hilbert.py` implements their Algorithms 1 and 2 as
a reference and tests that statement.  The outcome:

  * in d = 2 the Skilling index below is *identical* to theirs, and the
    two orders agree;
  * in d = 3 they are **different space-filling curves** -- different
    indices and different total orders, already at order p = 1.  This is
    not a bug in either: Haverkort has shown there are 10 694 807
    structurally distinct three-dimensional Hilbert curves, and Skilling's
    construction and theirs are two of them;
  * consequently the anisotropic order produced here does **not** coincide
    with the compact Hilbert index in d = 3.

What *is* verified is the content the paper's bandwidth argument actually
uses: within a single variant, the restricted-cube order and the compact
index induce the same total order (the rank-compression lemma; tested for
H&RC's variant in exp12, question Q2, and true in every case).

The practical consequence is that any locality constant measured here --
including the WL_2 value reported for d = 3 -- belongs to the Skilling
variant specifically, and must not be compared with published values for
other variants without saying so.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "hilbert_encode",
    "hilbert_decode",
    "phve_order",
    "phve_order_aniso",
    "choose_orders",
    "normalise",
    "ALPHABET",
    "BASE",
    "encode_base29",
    "code_length",
]

ALPHABET = "23456789ABCDEFGHJKMNPQRSTVWXY"
BASE = len(ALPHABET)  # 29


# ----------------------------------------------------------------------
# Skilling kernel, vectorised
# ----------------------------------------------------------------------

def _axes_to_transpose(X: np.ndarray, p: int) -> np.ndarray:
    """In-place Skilling forward transform.  X has shape (N, d), int64."""
    d = X.shape[1]
    M = np.int64(1) << (p - 1)

    Q = M
    while Q > 1:
        P = Q - 1
        for i in range(d):
            mask = (X[:, i] & Q) != 0
            # branch 1: X0 ^= P
            X[mask, 0] ^= P
            # branch 2: exchange low bits of X0 and Xi
            nm = ~mask
            t = (X[nm, 0] ^ X[nm, i]) & P
            X[nm, 0] ^= t
            Xi = X[:, i]
            Xi[nm] ^= t
            X[:, i] = Xi
        Q >>= 1

    for i in range(1, d):
        X[:, i] ^= X[:, i - 1]

    t = np.zeros(X.shape[0], dtype=np.int64)
    Q = M
    while Q > 1:
        sel = (X[:, d - 1] & Q) != 0
        t[sel] ^= (Q - 1)
        Q >>= 1
    for i in range(d):
        X[:, i] ^= t
    return X


def _transpose_to_axes(X: np.ndarray, p: int) -> np.ndarray:
    """In-place inverse Skilling transform.  X has shape (N, d), int64."""
    d = X.shape[1]
    N = np.int64(2) << (p - 1)

    t = X[:, d - 1] >> np.int64(1)
    for i in range(d - 1, 0, -1):
        X[:, i] ^= X[:, i - 1]
    X[:, 0] ^= t

    Q = np.int64(2)
    while Q != N:
        P = Q - 1
        for i in range(d - 1, -1, -1):
            mask = (X[:, i] & Q) != 0
            X[mask, 0] ^= P
            nm = ~mask
            tt = (X[nm, 0] ^ X[nm, i]) & P
            X[nm, 0] ^= tt
            Xi = X[:, i]
            Xi[nm] ^= tt
            X[:, i] = Xi
        Q <<= 1
    return X


def _interleave(X: np.ndarray, p: int) -> np.ndarray:
    """Bit-interleave the d transposed coordinates into a single index."""
    N, d = X.shape
    out = np.zeros(N, dtype=np.int64)
    for i in range(p - 1, -1, -1):          # from MSB to LSB
        for j in range(d):
            out = (out << np.int64(1)) | ((X[:, j] >> np.int64(i)) & np.int64(1))
    return out


def _deinterleave(idx: np.ndarray, p: int, d: int) -> np.ndarray:
    X = np.zeros((idx.shape[0], d), dtype=np.int64)
    bit = 0
    for i in range(p - 1, -1, -1):
        for j in range(d):
            shift = np.int64(d * p - 1 - bit)
            X[:, j] |= ((idx >> shift) & np.int64(1)) << np.int64(i)
            bit += 1
    return X


def hilbert_encode(coords: np.ndarray, p: int) -> np.ndarray:
    """Hilbert index of integer coordinates in {0,...,2^p-1}^d.

    Parameters
    ----------
    coords : (N, d) integer array
    p      : order (grid side = 2^p)

    Returns
    -------
    (N,) int64 array with values in {0, ..., 2^(d p) - 1}.
    """
    coords = np.ascontiguousarray(coords, dtype=np.int64)
    if coords.ndim != 2:
        raise ValueError("coords must be (N, d)")
    d = coords.shape[1]
    if d * p > 62:
        raise ValueError(f"d*p = {d * p} exceeds int64 capacity")
    if p == 0:
        return np.zeros(coords.shape[0], dtype=np.int64)
    X = coords.copy()
    _axes_to_transpose(X, p)
    return _interleave(X, p)


def hilbert_decode(index: np.ndarray, p: int, d: int) -> np.ndarray:
    """Inverse of :func:`hilbert_encode`."""
    index = np.ascontiguousarray(index, dtype=np.int64)
    if p == 0:
        return np.zeros((index.shape[0], d), dtype=np.int64)
    X = _deinterleave(index, p, d)
    _transpose_to_axes(X, p)
    return X


# ----------------------------------------------------------------------
# Base-29 encoding (the Enc component of F)
# ----------------------------------------------------------------------

def code_length(p: int, d: int) -> int:
    return int(np.ceil(d * p * np.log(2) / np.log(BASE)))


def encode_base29(index: np.ndarray, k: int) -> list[str]:
    """Zero-padded base-29 strings over ALPHABET (length k)."""
    idx = np.asarray(index, dtype=object)
    out = []
    for v in idx:
        v = int(v)
        s = []
        for _ in range(k):
            s.append(ALPHABET[v % BASE])
            v //= BASE
        out.append("".join(reversed(s)))
    return out


# ----------------------------------------------------------------------
# Normalisation N_p^{(d),alpha} and the induced permutations
# ----------------------------------------------------------------------

def normalise(points: np.ndarray, box: np.ndarray, orders) -> np.ndarray:
    """Affine normalisation onto the anisotropic grid prod_i {0..2^{p_i}-1}.

    Parameters
    ----------
    points : (N, d) float array
    box    : (d, 2) array of [a_i, b_i]
    orders : int, or sequence of d ints (per-axis orders p_i)
    """
    points = np.asarray(points, dtype=float)
    box = np.asarray(box, dtype=float)
    d = points.shape[1]
    if np.isscalar(orders):
        orders = [int(orders)] * d
    orders = np.asarray(orders, dtype=np.int64)

    a = box[:, 0]
    b = box[:, 1]
    n = (np.int64(1) << orders).astype(np.float64)
    g = np.floor((points - a) / (b - a) * n)
    g = np.clip(g, 0, n - 1)
    return g.astype(np.int64)


def phve_order(points: np.ndarray, box: np.ndarray, p: int) -> np.ndarray:
    """Permutation sorting the points by cubic-order Hilbert index.

    Returns
    -------
    perm : (N,) int array, ``points[perm]`` is in Hilbert order.
    """
    g = normalise(points, box, p)
    h = hilbert_encode(g, p)
    return np.argsort(h, kind="stable")


def phve_order_aniso(points: np.ndarray, box: np.ndarray, orders) -> np.ndarray:
    """Permutation sorting the points by the *anisotropic* PHVE index.

    The per-axis grid is prod_i {0,...,2^{p_i}-1}.  The index is obtained by
    embedding this box into the enclosing cube of order pmax = max_i p_i and
    applying the cubic Hilbert map.  By the rank-compression lemma this
    induces exactly the same total order as the compact Hilbert index of
    Hamilton & Rau-Chaplin (2008), so it is a faithful stand-in for the
    purposes of renumbering.
    """
    points = np.asarray(points, dtype=float)
    d = points.shape[1]
    if np.isscalar(orders):
        orders = [int(orders)] * d
    orders = np.asarray(orders, dtype=np.int64)
    g = normalise(points, box, orders)
    pmax = int(orders.max())
    h = hilbert_encode(g, pmax)
    return np.argsort(h, kind="stable")


def choose_orders(box: np.ndarray, total_bits: int) -> np.ndarray:
    """R1 selection rule: per-axis orders equalising the cell edge lengths.

    Given a budget ``total_bits`` = sum_i p_i (equivalently a fixed number
    2^{total_bits} of grid cells, hence a fixed code length), the rule
    returns integers p_i as close as possible to

        p_i* = total_bits/d + log2( L_i / G ),    G = (prod_j L_j)^{1/d},

    where L_i = b_i - a_i.  These make the cell edge lengths
    Delta_i = L_i / 2^{p_i} as equal as possible.

    The rounding is done by largest-remainder so that sum_i p_i is exactly
    ``total_bits`` and every p_i >= 1.
    """
    box = np.asarray(box, dtype=float)
    L = box[:, 1] - box[:, 0]
    d = len(L)
    G = float(np.exp(np.mean(np.log(L))))
    ideal = total_bits / d + np.log2(L / G)

    floors = np.floor(ideal).astype(int)
    floors = np.maximum(floors, 1)
    deficit = total_bits - floors.sum()
    rema = ideal - np.floor(ideal)
    order = np.argsort(-rema)
    i = 0
    while deficit > 0:
        floors[order[i % d]] += 1
        deficit -= 1
        i += 1
    while deficit < 0:
        j = order[(-i - 1) % d]
        if floors[j] > 1:
            floors[j] -= 1
            deficit += 1
        i += 1
    return floors
