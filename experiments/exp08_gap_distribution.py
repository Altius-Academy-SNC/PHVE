#!/usr/bin/env python3
"""
EXPERIMENT 08 -- The full distribution of index gaps.

The bandwidth bound of the previous version tried to describe a whole
distribution by its maximum, and got it wrong.  The claim under test here
is that the distribution itself has a clean, N-independent description:

    P( |i - j| > t )  <=  C t^{-1/d}      for the Hilbert numbering,

with C independent of N.  Everything else follows from it:

    median  = O(1)                          (P = 1/2 at t = O(1))
    mean    = int_0^N P(>t) dt = O(N^{1-1/d})
    max     = Theta(N)                      (the seam pairs)

The same measurement is made for reverse Cuthill-McKee, whose gaps should
instead concentrate around N^{(d-1)/d} with a sharp cut-off, and for the
natural ordering, which should be roughly uniform on [0, N].

The decisive graphical test is whether the Hilbert curves for different N
*collapse* onto a single curve when plotted as P(>t) against t; that is
what N-independence means.

Output: results/exp08_gap_distribution.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                      # noqa: E402
from phve.metrics import order_phve, order_rcm                   # noqa: E402
from common import FIGURES, hardware_record, save_json           # noqa: E402

SCHEMES = ("natural", "rcm", "phve")


def ccdf(gaps, ts):
    """Empirical P(gap > t) at the given thresholds."""
    g = np.sort(gaps)
    n = g.size
    idx = np.searchsorted(g, ts, side="right")
    return (n - idx) / n


def tail_exponent(ts, p, lo, hi):
    """Slope of log P(>t) against log t over [lo, hi]."""
    m = (ts >= lo) & (ts <= hi) & (p > 0)
    if m.sum() < 3:
        return None
    s, b = np.polyfit(np.log(ts[m]), np.log(p[m]), 1)
    resid = np.log(p[m]) - (s * np.log(ts[m]) + b)
    r2 = 1 - resid.var() / max(np.log(p[m]).var(), 1e-30)
    return {"exponent": float(s), "r2": float(r2),
            "range": [float(lo), float(hi)], "n_points": int(m.sum())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    ts = np.unique(np.round(np.logspace(0, 5.4, 90)).astype(np.int64))
    vol = model.load_mni152(2)
    records = []

    for n_target in args.sizes:
        run = model.DiffusionRun(vol, n_target=n_target, seed=args.seed,
                                 tau=args.tau)
        M, K, K0, g, gradnorm = run.operators()
        run.u, _ = fem.step(M, K, args.tau, run.u)
        run.remesh(gradnorm)
        edges = run.mesh.edges()
        n = run.mesh.n_vertices

        perms = {"natural": np.arange(n),
                 "rcm": order_rcm(run.mesh)[0],
                 "phve": order_phve(run.mesh, args.order)[0]}
        rec = {"N": int(n), "n_edges": int(edges.shape[0]),
               "n_target": n_target, "thresholds": ts.tolist()}
        for name, perm in perms.items():
            inv = np.empty(n, dtype=np.int64)
            inv[perm] = np.arange(n)
            d = np.abs(inv[edges[:, 0]] - inv[edges[:, 1]])
            p = ccdf(d, ts)
            rec[name] = {
                "ccdf": p.tolist(),
                "mean": float(d.mean()), "median": float(np.median(d)),
                "max": int(d.max()),
                "q": {str(q): float(np.percentile(d, q))
                      for q in (50, 75, 90, 95, 99, 99.9)},
                # fit the tail over a window that is inside the data for
                # every N of the sweep
                "tail_fit_10_1000": tail_exponent(ts.astype(float), p, 10, 1000),
                "tail_fit_30_3000": tail_exponent(ts.astype(float), p, 30, 3000),
            }
        records.append(rec)
        print("N=%6d | median: nat %7.0f rcm %6.0f phve %5.0f | "
              "tail exponent (10..1000): nat %6.3f rcm %6.3f phve %6.3f"
              % (n, rec["natural"]["median"], rec["rcm"]["median"],
                 rec["phve"]["median"],
                 rec["natural"]["tail_fit_10_1000"]["exponent"],
                 rec["rcm"]["tail_fit_10_1000"]["exponent"],
                 rec["phve"]["tail_fit_10_1000"]["exponent"]))

    # ---- collapse test: how much does the PHVE ccdf move with N? --------
    P = np.array([r["phve"]["ccdf"] for r in records])
    window = (ts >= 10) & (ts <= 1000)
    spread = {
        "max_abs_deviation_from_mean": float(
            np.abs(P[:, window] - P[:, window].mean(axis=0)).max()),
        "max_ratio_over_window": float(
            (P[:, window].max(axis=0) / np.maximum(P[:, window].min(axis=0), 1e-12)).max()),
        "N_range": [int(records[0]["N"]), int(records[-1]["N"])],
    }
    R = np.array([r["rcm"]["ccdf"] for r in records])
    spread_rcm = {
        "max_abs_deviation_from_mean": float(
            np.abs(R[:, window] - R[:, window].mean(axis=0)).max()),
    }

    print("\ncollapse of the PHVE ccdf over N in [%d, %d], t in [10, 1000]:"
          % tuple(spread["N_range"]))
    print("   max |P_N(t) - mean_N P(t)| = %.4f   (RCM, same window: %.4f)"
          % (spread["max_abs_deviation_from_mean"],
             spread_rcm["max_abs_deviation_from_mean"]))
    print("   predicted tail exponent -1/d = %.4f" % (-1.0 / 3.0))
    ex = [r["phve"]["tail_fit_10_1000"]["exponent"] for r in records]
    print("   measured PHVE tail exponents: " + ", ".join("%.3f" % e for e in ex))
    print("   mean %.4f, spread %.4f" % (float(np.mean(ex)),
                                         float(np.max(ex) - np.min(ex))))

    out = {"experiment": "exp08_gap_distribution", "params": vars(args),
           "hardware": hardware_record(), "records": records,
           "phve_ccdf_collapse": spread, "rcm_ccdf_collapse": spread_rcm,
           "predicted_tail_exponent": -1.0 / 3.0}
    save_json(out, "exp08_gap_distribution.json")

    # ---- figure ---------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        cmap = plt.get_cmap("viridis")
        for k, r in enumerate(records):
            c = cmap(k / max(len(records) - 1, 1))
            axes[0].loglog(ts, r["phve"]["ccdf"], color=c,
                           label=f"N={r['N']}")
            axes[1].loglog(ts, r["rcm"]["ccdf"], color=c, label=f"N={r['N']}")
        tt = np.array([10.0, 1000.0])
        ref = 0.5 * (tt / 10.0) ** (-1.0 / 3.0)
        axes[0].loglog(tt, ref, "k--", label=r"slope $-1/3$")
        for ax, title in zip(axes, ["PHVE (Hilbert)", "reverse Cuthill--McKee"]):
            ax.set_xlabel(r"$t$"); ax.set_ylabel(r"$P(|i-j|>t)$")
            ax.set_title(title); ax.legend(fontsize=7); ax.grid(alpha=.3)
        fig.tight_layout()
        path = os.path.join(FIGURES, "gap_distribution.png")
        fig.savefig(path, dpi=150)
        print(f"[saved] {path}")
    except Exception as exc:                      # pragma: no cover
        print(f"[figure skipped] {exc}")


if __name__ == "__main__":
    main()
