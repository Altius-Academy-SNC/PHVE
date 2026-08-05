#!/usr/bin/env python3
"""
EXPERIMENT 21 -- Is the Hilbert transposition of a *nonlocal* operator
comparable to a one-dimensional fractional operator?

Motivation
----------
Every attempt in this repository to make a space-filling curve help a
*local* operator has failed, and the reason is now structural rather than
accidental.  The Hilbert parametrisation gamma : [0,1] -> Omega is
1/d-Hoelder -- close in parameter implies close in space, quantified by
WL_2 -- but its inverse is Hoelder in no sense at all: spatially adjacent
points can sit Theta(n^d) apart in index (the seam theorem).  Transposing a
local operator needs the *inverse* direction, which is exactly the one that
does not exist.  And since gamma is measure preserving, transposition is an
isometry: it cannot reduce anything, which is the continuum shadow of the
equivalence lemma.

A nonlocal operator does not need the inverse direction.  For a kernel
K(x,y) = ||x - y||^{-(d+2s)}, the surviving bound
||gamma(a) - gamma(b)|| ~ |a - b|^{1/d} suggests

    K(gamma(a), gamma(b))  ~  |a - b|^{-(d+2s)/d}  =  |a - b|^{-1 - 2s/d},

i.e. a one-dimensional fractional kernel of order s/d.  If that holds up to
a set carrying little mass, the transposition maps a d-dimensional
fractional problem to a 1-D one plus a controlled perturbation, and there
is a genuine programme.  If the seam set carries a constant fraction of the
mass, there is not.

This script measures which it is.  Nothing is assumed: the comparison
constant c is fitted, and both the pointwise ratio and -- more importantly
-- the *row-sum* excess are reported, because operator bounds of Schur type
depend on row sums, not on pointwise ratios.

Definitions
-----------
For the grid of order p in dimension d, gamma(i) is the cell of Hilbert
rank i, rescaled to the unit cube.  Then

    Ktilde(i,j) = ||gamma(i) - gamma(j)||^{-(d+2s)}
    Model(i,j)  = c * |i - j|^{-(1 + 2s/d)},     c fitted by least squares
                                                 in log space on the bulk

  * pointwise ratio r(i,j) = Ktilde / Model;
  * bulk = pairs with r within [1/tol, tol];  seam = the rest;
  * row-sum mass fraction carried by the seam pairs, which is the quantity
    that decides whether the deviation is a perturbation or the main term.

Output: results/exp21_nonlocal_transposition.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.hilbert import hilbert_decode                            # noqa: E402
from common import hardware_record, save_json                      # noqa: E402


def positions(d, p):
    """gamma(i) for i = 0..2^{dp}-1, rescaled to [0,1)^d."""
    N = 2 ** (d * p)
    X = hilbert_decode(np.arange(N, dtype=np.int64), p, d).astype(float)
    return (X + 0.5) / (1 << p)


def analyse(d, p, s, tol, chunk=1024):
    X = positions(d, p)
    N = X.shape[0]
    expo_d = d + 2.0 * s              # kernel exponent in space
    expo_1 = 1.0 + 2.0 * s / d        # predicted 1-D exponent

    # accumulate log-ratio statistics and row sums in chunks
    sum_logdiff = 0.0
    n_pairs = 0
    row_total = np.zeros(N)
    row_seam = np.zeros(N)
    ratios_sample = []

    # first pass: fit c by matching means of log Ktilde - log |i-j|^{-expo_1}
    for s0 in range(0, N, chunk):
        e0 = min(s0 + chunk, N)
        D = X[s0:e0, None, :] - X[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", D, D))
        gap = np.abs(np.arange(s0, e0)[:, None] - np.arange(N)[None, :])
        m = gap > 0
        K = np.zeros_like(dist)
        K[m] = dist[m] ** (-expo_d)
        M0 = np.zeros_like(dist)
        M0[m] = gap[m].astype(float) ** (-expo_1)
        sum_logdiff += float(np.log(K[m]).sum() - np.log(M0[m]).sum())
        n_pairs += int(m.sum())
    log_c = sum_logdiff / max(n_pairs, 1)
    c = float(np.exp(log_c))

    # second pass: ratios, bulk/seam split, row sums
    for s0 in range(0, N, chunk):
        e0 = min(s0 + chunk, N)
        D = X[s0:e0, None, :] - X[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", D, D))
        gap = np.abs(np.arange(s0, e0)[:, None] - np.arange(N)[None, :])
        m = gap > 0
        K = np.zeros_like(dist)
        K[m] = dist[m] ** (-expo_d)
        M0 = np.zeros_like(dist)
        M0[m] = c * gap[m].astype(float) ** (-expo_1)
        r = np.ones_like(dist)
        r[m] = K[m] / M0[m]
        seam = m & ((r > tol) | (r < 1.0 / tol))
        row_total[s0:e0] = K.sum(axis=1)
        row_seam[s0:e0] = np.where(seam, K, 0.0).sum(axis=1)
        if s0 == 0:
            ratios_sample = r[m].ravel()[:200000].copy()

    frac_seam_pairs = None
    q = np.percentile(ratios_sample, [1, 5, 25, 50, 75, 95, 99])
    frac_seam_pairs = float(((ratios_sample > tol)
                             | (ratios_sample < 1.0 / tol)).mean())
    mass_frac = row_seam / np.maximum(row_total, 1e-300)

    return {
        "d": d, "p": p, "s": s, "N": int(N), "tol": tol,
        "fitted_c": c,
        "kernel_exponent_space": expo_d,
        "kernel_exponent_1d_predicted": expo_1,
        "ratio_percentiles_1_5_25_50_75_95_99": q.tolist(),
        "fraction_pairs_outside_tol": frac_seam_pairs,
        "seam_mass_fraction_mean": float(mass_frac.mean()),
        "seam_mass_fraction_median": float(np.median(mass_frac)),
        "seam_mass_fraction_max": float(mass_frac.max()),
        "seam_mass_fraction_p95": float(np.percentile(mass_frac, 95)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--p2", type=int, nargs="+", default=[4, 5, 6])
    ap.add_argument("--p3", type=int, nargs="+", default=[3, 4])
    ap.add_argument("--s", type=float, nargs="+", default=[0.25, 0.5, 0.75])
    ap.add_argument("--tol", type=float, default=2.0)
    args = ap.parse_args()

    out = {"experiment": "exp21_nonlocal_transposition", "params": vars(args),
           "hardware": hardware_record(), "rows": []}

    print("Comparing Ktilde(i,j) = ||gamma(i)-gamma(j)||^-(d+2s)")
    print("      against  c |i-j|^-(1+2s/d),  c fitted in log space.\n")
    print("  d  p    N     s   | ratio med |  q05    q95  | pairs off | "
          "row-mass in seam (mean / p95 / max)")
    for d in args.d:
        ps = args.p2 if d == 2 else args.p3
        for p in ps:
            for s in args.s:
                r = analyse(d, p, s, args.tol)
                out["rows"].append(r)
                q = r["ratio_percentiles_1_5_25_50_75_95_99"]
                print("  %d  %d %6d  %.2f | %9.3f | %6.3f %6.3f | %8.2f%% | "
                      "%.4f / %.4f / %.4f"
                      % (d, p, r["N"], s, q[3], q[1], q[5],
                         100 * r["fraction_pairs_outside_tol"],
                         r["seam_mass_fraction_mean"],
                         r["seam_mass_fraction_p95"],
                         r["seam_mass_fraction_max"]))
                sys.stdout.flush()

    print("\n=== Reading of the result ===")
    print("  The programme is viable only if the row-sum mass carried by")
    print("  pairs that deviate from the 1-D model stays SMALL and does not")
    print("  grow with N. If it is a constant fraction, the seam is the main")
    print("  term and the transposition buys nothing.")
    for d in args.d:
        rows = [r for r in out["rows"] if r["d"] == d]
        by_p = {}
        for r in rows:
            by_p.setdefault(r["p"], []).append(r["seam_mass_fraction_mean"])
        ps = sorted(by_p)
        print("   d=%d  mean seam mass by p: %s"
              % (d, ", ".join("p=%d: %.4f" % (p, float(np.mean(by_p[p])))
                              for p in ps)))

    save_json(out, "exp21_nonlocal_transposition.json")


if __name__ == "__main__":
    main()
