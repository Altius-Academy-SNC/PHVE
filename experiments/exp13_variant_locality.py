#!/usr/bin/env python3
"""
EXPERIMENT 13 -- Are the Skilling and Hamilton--Rau-Chaplin 3D curves the
same curve up to a symmetry of the cube, and if not, do they have the same
locality constants?

Why this matters
----------------
`exp12` established that in d = 3 our Skilling kernel and H&RC's standard
index are different total orders.  That leaves the decisive question open.

If the two orders differ only by an isometry of the cube -- one of the 48
signed axis permutations -- then they are the *same curve* seen in a
rotated frame.  Every locality measure is invariant under such a change, so
our measured WL_2 ~ 29.5 would have to equal the published 22.9, and the
discrepancy would signal an error on our side.

If instead they are genuinely different curves, then 29.5 and 22.9 are
values of different objects, were never comparable, and the only honest
statement is one that names the variant.

The script therefore does two things.

  (A) *Symmetry search.*  For each of the 48 signed axis permutations g of
      the cube, test whether the H&RC index composed with g induces the
      same total order as the Skilling index.  Exhaustive over all points,
      for p = 1..4.  Also tested with the order reversed (i -> n^d-1-i),
      which is a symmetry of the traversal, giving 96 candidates.

  (B) *Locality constants of both variants.*  Computed identically for
      each, so the comparison is like for like:

        A(p)   = max over spatially adjacent pairs of |i - j|
                 (the forward/seam measure)
        WL_2   = max over pairs of ||x - y||_2^d / |i - j|
                 (the inverse measure; this is the quantity the literature
                 calls worst-case L2 dilation)

      Exhaustive over all pairs for p <= 4.  For p = 5 a gap-restricted
      sweep is used, with the restriction reported so the reader knows it
      is a lower bound on the maximum.

Output: results/exp13_variant_locality.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.compact_hilbert import hilbert_index_hrc              # noqa: E402
from phve.hilbert import hilbert_encode                          # noqa: E402
from common import hardware_record, save_json                    # noqa: E402


def full_cube(d, p):
    n = 1 << p
    axes = [np.arange(n, dtype=np.int64)] * d
    return np.array(list(itertools.product(*axes)), dtype=np.int64)


def hrc_index_array(G, d, p):
    return np.array([hilbert_index_hrc(row, d, p) for row in G],
                    dtype=np.int64)


def cube_symmetries(d):
    """All signed axis permutations: (perm, flips)."""
    for perm in itertools.permutations(range(d)):
        for flips in itertools.product([False, True], repeat=d):
            yield perm, flips


def apply_symmetry(G, perm, flips, p):
    n = (1 << p) - 1
    H = G[:, list(perm)].copy()
    for j, f in enumerate(flips):
        if f:
            H[:, j] = n - H[:, j]
    return H


def adjacent_pairs(G, d, p):
    """Indices of all pairs of points at unit Euclidean distance."""
    n = 1 << p
    lin = np.zeros(G.shape[0], dtype=np.int64)
    for j in range(d):
        lin = lin * n + G[:, j]
    pos = np.empty(n ** d, dtype=np.int64)
    pos[lin] = np.arange(G.shape[0])
    pairs = []
    for j in range(d):
        keep = G[:, j] < n - 1
        src = np.nonzero(keep)[0]
        Gn = G[src].copy()
        Gn[:, j] += 1
        lin2 = np.zeros(Gn.shape[0], dtype=np.int64)
        for k in range(d):
            lin2 = lin2 * n + Gn[:, k]
        pairs.append(np.stack([src, pos[lin2]], axis=1))
    return np.concatenate(pairs, axis=0)


def forward_max(idx, pairs):
    return int(np.abs(idx[pairs[:, 0]] - idx[pairs[:, 1]]).max())


def wl2_exhaustive(G, idx, d, chunk=512):
    """max ||x-y||_2^d / |i-j| over all pairs.  O(N^2), chunked."""
    order = np.argsort(idx)
    P = G[order].astype(np.float64)          # position of rank r
    N = P.shape[0]
    best = 0.0
    arg = None
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        # distances from ranks [s,e) to all ranks > current
        D = P[s:e][:, None, :] - P[None, :, :]
        dist2 = np.einsum("ijk,ijk->ij", D, D)
        gap = np.abs(np.arange(s, e)[:, None] - np.arange(N)[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            val = np.power(dist2, d / 2.0) / gap
        val[gap == 0] = 0.0
        m = float(np.nanmax(val))
        if m > best:
            best = m
            i, j = np.unravel_index(np.nanargmax(val), val.shape)
            arg = (int(s + i), int(j))
    return best, arg


def wl2_restricted(G, idx, d, max_gap):
    """max over pairs with |i-j| <= max_gap.  A lower bound on WL_2."""
    order = np.argsort(idx)
    P = G[order].astype(np.float64)
    N = P.shape[0]
    best = 0.0
    for g in range(1, max_gap + 1):
        D = P[g:] - P[:-g]
        dist2 = np.einsum("ij,ij->i", D, D)
        v = float(np.power(dist2, d / 2.0).max() / g)
        if v > best:
            best = v
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=3)
    ap.add_argument("--sym-orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--loc-orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--restricted-order", type=int, default=5)
    ap.add_argument("--gap-fraction", type=float, default=0.08)
    args = ap.parse_args()

    d = args.d
    out = {"experiment": "exp13_variant_locality", "params": vars(args),
           "hardware": hardware_record()}

    # ------------------------------------------------------------------
    # (A) Symmetry search
    # ------------------------------------------------------------------
    print("=== (A) Is Skilling = H&RC composed with a cube symmetry? ===")
    sym_rows = []
    for p in args.sym_orders:
        G = full_cube(d, p)
        sk = hilbert_encode(G, p)
        sk_rank = np.argsort(np.argsort(sk))
        matches = []
        for perm, flips in cube_symmetries(d):
            Gs = apply_symmetry(G, perm, flips, p)
            hr = hrc_index_array(Gs, d, p)
            hr_rank = np.argsort(np.argsort(hr))
            if np.array_equal(sk_rank, hr_rank):
                matches.append({"perm": list(perm), "flips": list(flips),
                                "reversed": False})
            if np.array_equal(sk_rank, (len(G) - 1 - hr_rank)):
                matches.append({"perm": list(perm), "flips": list(flips),
                                "reversed": True})
        sym_rows.append({"p": p, "n_points": int(len(G)),
                         "n_matches": len(matches), "matches": matches})
        print("   p=%d  %6d points  matching symmetries: %d"
              % (p, len(G), len(matches)))
        sys.stdout.flush()
    out["symmetry_search"] = sym_rows
    any_sym = any(r["n_matches"] > 0 for r in sym_rows)
    all_sym = all(r["n_matches"] > 0 for r in sym_rows)
    print("   -> some order admits a symmetry: %s ; every order: %s"
          % (any_sym, all_sym))

    # ------------------------------------------------------------------
    # (B) Locality constants, both variants, same code
    # ------------------------------------------------------------------
    print("\n=== (B) Locality constants of the two variants ===")
    print("   p |    A_skilling  A_hrc |   WL2_skilling  WL2_hrc")
    loc_rows = []
    for p in args.loc_orders:
        G = full_cube(d, p)
        pairs = adjacent_pairs(G, d, p)
        sk = hilbert_encode(G, p)
        hr = hrc_index_array(G, d, p)
        a_sk = forward_max(sk, pairs)
        a_hr = forward_max(hr, pairs)
        w_sk, arg_sk = wl2_exhaustive(G, sk, d)
        w_hr, arg_hr = wl2_exhaustive(G, hr, d)
        loc_rows.append({"p": p, "n_points": int(len(G)),
                         "A_skilling": a_sk, "A_hrc": a_hr,
                         "WL2_skilling": w_sk, "WL2_hrc": w_hr,
                         "WL2_equal": abs(w_sk - w_hr) < 1e-12,
                         "A_equal": a_sk == a_hr})
        print("   %d | %13d %6d | %14.4f %8.4f"
              % (p, a_sk, a_hr, w_sk, w_hr))
        sys.stdout.flush()
    out["locality"] = loc_rows

    # restricted sweep one order further
    p = args.restricted_order
    if p:
        G = full_cube(d, p)
        N = len(G)
        mg = max(1, int(args.gap_fraction * N))
        sk = hilbert_encode(G, p)
        hr = hrc_index_array(G, d, p)
        w_sk = wl2_restricted(G, sk, d, mg)
        w_hr = wl2_restricted(G, hr, d, mg)
        pairs = adjacent_pairs(G, d, p)
        rec = {"p": p, "n_points": int(N), "max_gap_swept": mg,
               "gap_fraction": args.gap_fraction,
               "A_skilling": forward_max(sk, pairs),
               "A_hrc": forward_max(hr, pairs),
               "WL2_skilling_lower": w_sk, "WL2_hrc_lower": w_hr}
        out["locality_restricted"] = rec
        print("   %d | %13d %6d | %14.4f %8.4f   (gap sweep <= %d, lower bounds)"
              % (p, rec["A_skilling"], rec["A_hrc"], w_sk, w_hr, mg))

    # ------------------------------------------------------------------
    print("\n=== Verdict ===")
    verdict = {
        "same_curve_up_to_cube_symmetry": all_sym,
        "A_identical_all_orders": all(r["A_equal"] for r in loc_rows),
        "WL2_identical_all_orders": all(r["WL2_equal"] for r in loc_rows),
    }
    for k, v in verdict.items():
        print("   %-32s %s" % (k, v))
    if not verdict["same_curve_up_to_cube_symmetry"]:
        print("\n   The two are genuinely different curves in d = 3.")
        print("   Locality constants measured here are properties of the")
        print("   Skilling variant and must be labelled as such; published")
        print("   values for other variants are not comparable.")
    out["verdict"] = verdict

    save_json(out, "exp13_variant_locality.json")


if __name__ == "__main__":
    main()
