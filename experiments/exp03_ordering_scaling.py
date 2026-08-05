#!/usr/bin/env python3
"""
EXPERIMENT 03 -- Scaling of the index-gap functionals with N.

This is the experiment that decides the corrected form of the bandwidth
theorem.  For a family of adaptive meshes of growing size we measure, for
the three orderings, the whole distribution of

    D_ij = |i - j|   over the non-zeros of the assembled matrix,

not just its maximum.  The claims under test are

    BW_max(Hilbert) = Theta(N)              (the seam obstruction)
    BW_max(RCM)     = O(N^{(d-1)/d})
    E[D] (Hilbert)  = O(N^{(d-1)/d})        (corrected positive result)

Output: results/exp03_ordering_scaling.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                      # noqa: E402
from phve.metrics import order_phve, order_rcm                   # noqa: E402
from common import hardware_record, save_json                    # noqa: E402


def gap_stats(rows, cols, perm, n):
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    d = np.abs(inv[rows] - inv[cols])
    off = d[d > 0]
    if off.size == 0:
        off = d
    return {
        "max": int(d.max()),
        "mean": float(d.mean()),
        "median": float(np.median(d)),
        "p90": float(np.percentile(d, 90)),
        "p99": float(np.percentile(d, 99)),
        "p999": float(np.percentile(d, 99.9)),
        "rms": float(np.sqrt((d.astype(float) ** 2).mean())),
        "frac_gt_sqrt_n": float((d > np.sqrt(n)).mean()),
        "frac_gt_n_over_10": float((d > n / 10).mean()),
        "count_gt_n_over_4": int((d > n / 4).sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2000, 4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    records = []
    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed,
                                     tau=args.tau)
            # one diffusion step, then one adaptive remesh, so the mesh
            # measured is genuinely graded and not the initial cloud
            M, K, K0, g, gradnorm = run.operators()
            run.u, _ = fem.step(M, K, run.tau, run.u)
            run.remesh(gradnorm)
            M, K, K0, g, gradnorm = run.operators()
            A = (M + args.tau * K).tocoo()
            n = A.shape[0]
            rows = A.row.astype(np.int64)
            cols = A.col.astype(np.int64)

            perm_r, t_r = order_rcm(run.mesh)
            perm_h, t_h = order_phve(run.mesh, args.order)
            h_edges = run.mesh.edge_lengths()
            L = run.mesh.box[:, 1] - run.mesh.box[:, 0]

            rec = {
                "seed": seed, "n_target": n_target, "N": int(n),
                "n_tets": run.mesh.n_tets, "nnz": int(A.nnz),
                "h_median_mm": float(np.median(h_edges)),
                "h_median_over_L": float(np.median(h_edges) / L.max()),
                "N_pow_2_3": float(n ** (2.0 / 3.0)),
                "reorder_seconds": {"rcm": t_r, "phve": t_h},
                "natural": gap_stats(rows, cols, np.arange(n), n),
                "rcm": gap_stats(rows, cols, perm_r, n),
                "phve": gap_stats(rows, cols, perm_h, n),
            }
            records.append(rec)
            print("seed=%d N=%6d | max: nat %6d rcm %6d phve %6d "
                  "| mean: nat %7.1f rcm %6.1f phve %6.1f | N^{2/3}=%6.1f"
                  % (seed, n, rec["natural"]["max"], rec["rcm"]["max"],
                     rec["phve"]["max"], rec["natural"]["mean"],
                     rec["rcm"]["mean"], rec["phve"]["mean"],
                     rec["N_pow_2_3"]))

    # least-squares fits of log(metric) vs log(N)
    fits = {}
    N = np.array([r["N"] for r in records], dtype=float)
    for scheme in ("natural", "rcm", "phve"):
        for metric in ("max", "mean", "median", "p99"):
            y = np.array([r[scheme][metric] for r in records], dtype=float)
            ok = y > 0
            if ok.sum() < 2:
                continue
            slope, intercept = np.polyfit(np.log(N[ok]), np.log(y[ok]), 1)
            resid = np.log(y[ok]) - (slope * np.log(N[ok]) + intercept)
            fits[f"{scheme}.{metric}"] = {
                "exponent": float(slope),
                "prefactor": float(np.exp(intercept)),
                "r2": float(1 - resid.var() / np.log(y[ok]).var()),
            }

    out = {
        "experiment": "exp03_ordering_scaling",
        "params": vars(args),
        "hardware": hardware_record(),
        "records": records,
        "fits_metric_vs_N": fits,
        "reference_exponents": {"Theta(N)": 1.0, "Theta(N^{2/3})": 2.0 / 3.0},
    }
    save_json(out, "exp03_ordering_scaling.json")

    print("\nfitted exponents  metric ~ N^alpha")
    for k, v in sorted(fits.items()):
        print("   %-16s alpha = %6.3f   (R^2 = %.4f)" % (k, v["exponent"], v["r2"]))


if __name__ == "__main__":
    main()
