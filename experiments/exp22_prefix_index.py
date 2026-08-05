#!/usr/bin/env python3
"""
EXPERIMENT 22 -- Prefix range queries against a spatial index.

What is being compared, and what is not
---------------------------------------
A PHVE prefix identifies exactly one *dyadic cell* of the reference volume.
By the truncation theorem the codes inside a dyadic cell form a contiguous
range of Hilbert indices, so on a sorted array -- or equivalently on any
ordered index, a B-tree, a database column with an index on it -- the query
"all points in this cell" is two binary searches and a slice.

That is a **restricted query class**.  A kd-tree answers arbitrary boxes and
arbitrary balls; the prefix index answers dyadic cells only.  Comparing
them on arbitrary boxes would be meaningless.  The comparison here is
therefore on the class the encoding is designed for, and the claim under
test is narrow and practical:

    for dyadic-cell queries, no spatial index is needed -- an ordinary
    ordered index over an integer column is enough, and it is cheaper to
    build and smaller to store.

Both methods are checked to return **identical point sets** before any
timing is reported; a speed comparison between two methods that disagree
would be worthless.

Measured: index construction time, index memory, query time over many
random dyadic cells at several truncation depths, and the selectivity of
each query (so the reader can see the timings are not dominated by output
size).

Output: results/exp22_prefix_index.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "simulations"))

from phve import model                                             # noqa: E402
from phve.hilbert import hilbert_encode                            # noqa: E402
from common import hardware_record, save_json                      # noqa: E402
import codec3d                                                      # noqa: E402


def brain_voxels_mm(resolution=2):
    from nilearn.datasets import load_mni152_brain_mask
    vol = model.load_mni152(resolution)
    mask = np.asarray(load_mni152_brain_mask(resolution=resolution).dataobj
                      ).astype(bool)
    idx = np.argwhere(mask)
    return idx @ vol.affine[:3, :3].T + vol.affine[:3, 3]


def to_grid(xyz_mm, p, volume="CR"):
    dims = np.asarray(codec3d.VOLUMES[volume]["dims"], float)
    n = 1 << p
    g = np.floor((np.asarray(xyz_mm, float) + dims / 2.0) / dims * n)
    return np.clip(g, 0, n - 1).astype(np.int64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=8)
    ap.add_argument("--volume", type=str, default="CR")
    ap.add_argument("--depths", type=int, nargs="+", default=[2, 3, 4, 5],
                    help="dyadic level of the queried cell (cell = 2^-depth "
                         "of the box along each axis)")
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    xyz = brain_voxels_mm(2)
    N = xyz.shape[0]
    g = to_grid(xyz, args.p, args.volume)
    print("brain voxels: %d, grid order p = %d" % (N, args.p))

    # ---- build the two indices ---------------------------------------
    t0 = time.perf_counter()
    h = hilbert_encode(g, args.p)
    order = np.argsort(h, kind="stable")
    h_sorted = h[order]
    t_build_prefix = time.perf_counter() - t0
    mem_prefix = h_sorted.nbytes + order.nbytes

    from scipy.spatial import cKDTree
    t0 = time.perf_counter()
    tree = cKDTree(g.astype(float))
    t_build_tree = time.perf_counter() - t0
    # cKDTree memory is not directly exposed; report the point array, which
    # is a lower bound, and say so.
    mem_tree_lower = g.astype(float).nbytes

    print("build: prefix index %.4f s (%.1f MiB)  |  kd-tree %.4f s "
          "(>= %.1f MiB of points)"
          % (t_build_prefix, mem_prefix / 2 ** 20, t_build_tree,
             mem_tree_lower / 2 ** 20))

    rows = []
    for depth in args.depths:
        if depth > args.p:
            continue
        shift = args.p - depth               # bits below the cell
        cell = 1 << shift
        # candidate non-empty cells, so queries are not mostly empty
        cell_id = (g >> shift)
        uniq = np.unique(cell_id, axis=0)
        pick = uniq[rng.choice(uniq.shape[0],
                               size=min(args.queries, uniq.shape[0]),
                               replace=False)]

        # The client supplies a *code prefix*, so the Hilbert index of the
        # cell is not computed at query time: it is read off the prefix.
        # Precomputing it here mirrors that, and the timing below covers
        # only what a query actually costs -- the two binary searches.
        block = 1 << (3 * shift)
        hb = hilbert_encode(pick.astype(np.int64), depth)
        starts = hb.astype(np.int64) * block

        t_pref = []
        t_kd = []
        sels = []
        mismatches = 0
        for c, start in zip(pick, starts):
            lo_g = c * cell
            hi_g = lo_g + cell            # half-open

            # --- prefix query: one contiguous range of Hilbert indices ---
            t0 = time.perf_counter()
            i0 = np.searchsorted(h_sorted, start, side="left")
            i1 = np.searchsorted(h_sorted, start + block, side="left")
            res_prefix = order[i0:i1]
            t_pref.append(time.perf_counter() - t0)

            # --- kd-tree box query ---------------------------------------
            # A Chebyshev ball *is* the axis-aligned box, so this is exact
            # and needs no post-filter; the epsilon guards against a corner
            # point landing exactly on the radius.
            t0 = time.perf_counter()
            centre = (lo_g + hi_g - 1) / 2.0
            half = (cell - 1) / 2.0
            cand = tree.query_ball_point(centre, r=half + 1e-9, p=np.inf)
            res_tree = np.asarray(cand, dtype=np.int64)
            t_kd.append(time.perf_counter() - t0)

            if not np.array_equal(np.sort(res_prefix), np.sort(res_tree)):
                mismatches += 1
            sels.append(res_prefix.size)

        rec = {"depth": depth, "cell_voxels_side": int(cell),
               "n_queries": int(pick.shape[0]),
               "mismatches": int(mismatches),
               "selectivity_mean": float(np.mean(sels)),
               "selectivity_median": float(np.median(sels)),
               "prefix_ms_median": float(np.median(t_pref) * 1e3),
               "kdtree_ms_median": float(np.median(t_kd) * 1e3),
               "speedup_median": float(np.median(t_kd) / max(np.median(t_pref), 1e-12))}
        rows.append(rec)
        print("  depth=%d cell=%3d voxels/side | %3d queries | mismatches %d"
              " | median selectivity %7.0f | prefix %.4f ms  kd-tree %.4f ms"
              "  (x%.1f)"
              % (depth, cell, rec["n_queries"], mismatches,
                 rec["selectivity_median"], rec["prefix_ms_median"],
                 rec["kdtree_ms_median"], rec["speedup_median"]))
        sys.stdout.flush()

    total_mismatch = sum(r["mismatches"] for r in rows)
    print("\n=== Correctness ===")
    print("  total queries with differing result sets: %d" % total_mismatch)
    if total_mismatch:
        print("  !! timings below are meaningless until this is zero")

    out = {"experiment": "exp22_prefix_index", "params": vars(args),
           "hardware": hardware_record(), "n_voxels": int(N),
           "build": {"prefix_s": t_build_prefix,
                     "prefix_bytes": int(mem_prefix),
                     "kdtree_s": t_build_tree,
                     "kdtree_points_bytes_lower_bound": int(mem_tree_lower)},
           "queries": rows, "total_mismatches": int(total_mismatch)}
    save_json(out, "exp22_prefix_index.json")


if __name__ == "__main__":
    main()
