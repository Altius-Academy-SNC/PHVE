#!/usr/bin/env python3
"""
EXPERIMENT 11 -- The constants of the mean index-gap theorem (closes U8).

`exp03_ordering_scaling.py` tested only the *exponent* of

    (1/|E|) sum_{e in E} |pi(x) - pi(y)|
        <= 2 D c1 c2 / (1 - 2^{1-d}) * (N^2/|E|) * h * sum_i 1/L_i  +  c1,

measuring 0.643 against the predicted 2/3.  The constants c1 (dyadic
density, A3), c2 (slab density, A4) and D (maximum degree, A2) were never
estimated, so the *prefactor* of the bound was untested and the theorem
could have been vacuous on the meshes we actually use: a bound with a
constant of 10^4 says nothing about a measured mean gap of 10^3.

This script measures all four quantities directly from the mesh --

    D  = max degree of the mesh graph,
    h  = max Euclidean edge length,
    c1 = max over dyadic levels l and cells C of level l of
             #(V cap C) * 2^{dl} / N,
    c2 = max over axes i of  max_t #{x : |x_i - t| <= h} * L_i / (N h),

the last by an exact sliding window over the sorted coordinates, which
attains the supremum over t because the count is piecewise constant and
right-continuous with jumps only at x_j - h.

It then evaluates the bound and reports the ratio bound/measured.  A ratio
>= 1 on every mesh is what makes the theorem non-vacuous; the size of the
ratio is what makes it useful, and is reported honestly either way.

Output: results/exp11_meangap_constants.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import model                                            # noqa: E402
from phve.hilbert import normalise                                # noqa: E402
from phve.metrics import order_phve, order_rcm                    # noqa: E402
from common import hardware_record, save_json                     # noqa: E402


def max_degree(edges, n):
    deg = np.bincount(edges[:, 0], minlength=n) + \
          np.bincount(edges[:, 1], minlength=n)
    return int(deg.max()), deg


def dyadic_constant(points, box, levels, N):
    """c1 and the per-level occupancies.

    A dyadic cell of level l is one of the 2^{dl} cells of the uniform
    subdivision of the box.  `normalise` maps a point to its cell index at
    order l, so counting is a group-by on the linearised cell index.
    """
    per_level = []
    c1 = 0.0
    d = points.shape[1]
    for lev in levels:
        g = normalise(points, box, lev)                    # (N, d) in [0,2^lev)
        key = np.zeros(g.shape[0], dtype=np.int64)
        for j in range(d):
            key = (key << np.int64(lev)) | g[:, j]
        _, counts = np.unique(key, return_counts=True)
        occ = int(counts.max())
        val = occ * (2.0 ** (d * lev)) / N
        per_level.append({"level": int(lev), "max_occupancy": occ,
                          "n_nonempty_cells": int(counts.size),
                          "c1_level": val})
        c1 = max(c1, val)
    return max(c1, 1.0), per_level


def slab_constant(points, box, h, N):
    """c2 = max_i max_t #{|x_i - t| <= h} * L_i / (N h), exactly.

    For a fixed axis the count as a function of t is maximised at some t
    with t - h equal to one of the coordinates, so sweeping the sorted
    coordinates with a window of width 2h attains the supremum.
    """
    box = np.asarray(box, float)
    L = box[:, 1] - box[:, 0]
    d = points.shape[1]
    per_axis = []
    c2 = 0.0
    for i in range(d):
        x = np.sort(points[:, i])
        # window [x_j, x_j + 2h] for every j; count by binary search
        hi = np.searchsorted(x, x + 2.0 * h, side="right")
        counts = hi - np.arange(x.size)
        m = int(counts.max())
        val = m * L[i] / (N * h)
        per_axis.append({"axis": int(i), "max_slab_count": m,
                         "L_i": float(L[i]), "c2_axis": float(val)})
        c2 = max(c2, val)
    return max(c2, 1.0), per_axis


def mean_gap(edges, perm):
    inv = np.empty(perm.size, dtype=np.int64)
    inv[perm] = np.arange(perm.size)
    g = np.abs(inv[edges[:, 0]].astype(np.int64)
               - inv[edges[:, 1]].astype(np.int64))
    return float(g.mean()), float(np.median(g)), int(g.max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000, 128000])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--max-level", type=int, default=None,
                    help="highest dyadic level for c1 (default: --order)")
    args = ap.parse_args()

    vol = model.load_mni152(2)
    d = 3
    records = []

    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed)
            mesh = run.mesh
            pts, box = mesh.points, np.asarray(mesh.box, float)
            N = mesh.n_vertices
            edges = mesh.edges()
            E = edges.shape[0]

            D, deg = max_degree(edges, N)
            h = float(mesh.edge_lengths().max())
            L = box[:, 1] - box[:, 0]
            S = float(np.sum(1.0 / L))

            top = args.max_level if args.max_level else args.order
            levels = list(range(1, top + 1))
            c1, per_level = dyadic_constant(pts, box, levels, N)
            c2, per_axis = slab_constant(pts, box, h, N)

            bound = (2.0 * D * c1 * c2 / (1.0 - 2.0 ** (1 - d))
                     * (N ** 2 / E) * h * S) + c1

            # --- the repair -------------------------------------------
            # (A3) as published is required at *every* level. It cannot
            # hold with an N-independent constant: as soon as 2^{d l} > N a
            # non-empty cell holds >= 1 vertex while the bound c1 N 2^{-dl}
            # is < 1, forcing c1 >= 2^{dl}/N. So c1 is pinned to the finest
            # level and grows like 2^{dp}/N, which is what makes the bound
            # vacuous. Restricting (A3) to levels l <= l* with 2^{d l*} <= N
            # -- and truncating the sum in the proof at l*, which is legal
            # because two points sharing a level-l cell with l > l* also
            # share the level-l* cell -- gives a genuine mesh constant.
            l_star = max(1, int(np.floor(np.log2(N) / d)))
            adm = [r for r in per_level if r["level"] <= l_star]
            c1_star = max(1.0, max(r["c1_level"] for r in adm)) if adm else 1.0
            bound_star = (2.0 * D * c1_star * c2 / (1.0 - 2.0 ** (1 - d))
                          * (N ** 2 / E) * h * S) + c1_star

            perm_h, _ = order_phve(mesh, args.order)
            perm_r, _ = order_rcm(mesh)
            mh, medh, maxh = mean_gap(edges, perm_h)
            mr, medr, maxr = mean_gap(edges, perm_r)

            rec = {"seed": seed, "N": int(N), "E": int(E),
                   "D_max_degree": D, "mean_degree": float(deg.mean()),
                   "h_max_edge": h, "L": L.tolist(), "S_sum_inv_L": S,
                   "c1": float(c1), "c2": float(c2),
                   "c1_per_level": per_level, "c2_per_axis": per_axis,
                   "l_star": int(l_star), "c1_restricted": float(c1_star),
                   "bound_mean_gap_restricted": float(bound_star),
                   "ratio_restricted_over_measured": None,
                   "bound_mean_gap": float(bound),
                   "measured_mean_gap_phve": mh,
                   "measured_median_gap_phve": medh,
                   "measured_max_gap_phve": maxh,
                   "measured_mean_gap_rcm": mr,
                   "ratio_bound_over_measured": float(bound / mh),
                   # the same bound with all constants set to 1, i.e. the
                   # part that is genuinely N-dependent
                   "bound_constants_one": float(
                       2.0 / (1.0 - 2.0 ** (1 - d)) * (N ** 2 / E) * h * S + 1.0),
                   }
            rec["ratio_restricted_over_measured"] = float(bound_star / mh)
            records.append(rec)
            print("seed=%d N=%6d E=%7d | D=%3d h=%.2f c2=%.2f | published:"
                  " c1=%11.1f ratio %.3e | repaired (l*=%d): c1=%6.2f"
                  " bound %.3e ratio %8.1f"
                  % (seed, N, E, D, h, c2, c1, bound / mh, l_star, c1_star,
                     bound_star, bound_star / mh))
            sys.stdout.flush()

    print("\n=== Constants across the mesh sequence ===")
    print("  c1 : min %.2f  max %.2f" % (min(r["c1"] for r in records),
                                         max(r["c1"] for r in records)))
    print("  c2 : min %.2f  max %.2f" % (min(r["c2"] for r in records),
                                         max(r["c2"] for r in records)))
    print("  D  : min %d  max %d" % (min(r["D_max_degree"] for r in records),
                                     max(r["D_max_degree"] for r in records)))
    ratios = [r["ratio_bound_over_measured"] for r in records]
    print("  bound/measured : min %.1f  max %.1f" % (min(ratios), max(ratios)))

    # Is c1 bounded in N, as (A3) requires with a constant independent of N?
    print("\n=== N-dependence of the constants (the hypothesis at stake) ===")
    print("  c1 as published is pinned to the finest level: it should equal")
    print("  (max occupancy at level p) * 2^{dp} / N, hence grow like 1/N.")
    for r in sorted(records, key=lambda x: (x["seed"], x["N"])):
        pred = (2.0 ** (d * args.order)) / r["N"]
        print("  seed=%d N=%6d  c1=%11.1f  (2^{dp}/N = %11.1f, ratio %.2f)"
              "  c1*=%6.2f  l*=%d"
              % (r["seed"], r["N"], r["c1"], pred, r["c1"] / pred,
                 r["c1_restricted"], r["l_star"]))

    print("\n=== The repaired constant and bound ===")
    rr = [r["ratio_restricted_over_measured"] for r in records]
    c1s = [r["c1_restricted"] for r in records]
    print("  c1* : min %.2f  max %.2f   (bounded in N: %s)"
          % (min(c1s), max(c1s), max(c1s) / min(c1s) < 3.0))
    print("  repaired bound / measured : min %.1f  max %.1f" % (min(rr), max(rr)))
    print("  improvement over the published constant: x%.3e"
          % (np.mean(ratios) / np.mean(rr)))

    out = {"experiment": "exp11_meangap_constants", "params": vars(args),
           "hardware": hardware_record(), "records": records}
    save_json(out, "exp11_meangap_constants.json")


if __name__ == "__main__":
    main()
