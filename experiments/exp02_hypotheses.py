#!/usr/bin/env python3
"""
EXPERIMENT 02 -- Verification of the hypotheses of the published theorems
on the model problem.

(a) "N <= n^d, at most one node per cell".  Measured as a collision rate
    on the graded adaptive meshes produced by the model problem, together
    with the smallest order p that makes it hold.
(b) Validity regime of the bandwidth bound: h/Delta along the refinement
    history, and the point where (h/Delta)^d << N ceases to hold.
(c) Anisotropy: the discrepancy between grid distance and physical
    distance as a function of kappa = max_i L_i / min_i L_i.
(d) Encode/decode round trip on the *adaptive tetrahedral meshes*, not
    merely on a voxel grid, checked against the quantisation bound.

Output: results/exp02_hypotheses.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import model                                          # noqa: E402
from phve.hilbert import (hilbert_decode, hilbert_encode,        # noqa: E402
                          normalise)
from common import hardware_record, save_json                    # noqa: E402


# ----------------------------------------------------------------------

def collision_report(points, box, orders_range):
    """(a) how many mesh nodes land in the same PHVE cell."""
    rows = []
    N = points.shape[0]
    for p in orders_range:
        g = normalise(points, box, p)
        key = g[:, 0] * (1 << (2 * p)) + g[:, 1] * (1 << p) + g[:, 2]
        _, counts = np.unique(key, return_counts=True)
        n_cells_used = counts.size
        collided = int(N - n_cells_used)
        rows.append({
            "p": int(p),
            "n_cells_grid": float(2.0 ** (3 * p)),
            "N": int(N),
            "N_le_n_d": bool(N <= 2.0 ** (3 * p)),
            "distinct_cells": int(n_cells_used),
            "colliding_nodes": collided,
            "collision_rate": collided / N,
            "max_nodes_per_cell": int(counts.max()),
        })
    return rows


def hdelta_report(points, box, edges, p):
    """(b) h/Delta statistics for one mesh at order p."""
    L = box[:, 1] - box[:, 0]
    Delta_max = L.max() / 2 ** p
    Delta_min = L.min() / 2 ** p
    h = np.linalg.norm(points[edges[:, 0]] - points[edges[:, 1]], axis=1)
    N = points.shape[0]
    hmax = float(h.max())
    return {
        "p": int(p),
        "N": int(N),
        "Delta_max_mm": float(Delta_max),
        "Delta_min_mm": float(Delta_min),
        "h_min_mm": float(h.min()),
        "h_median_mm": float(np.median(h)),
        "h_max_mm": float(hmax),
        "h_over_Delta_median": float(np.median(h) / Delta_max),
        "h_over_Delta_max": float(hmax / Delta_max),
        "bound_term_median": float((np.median(h) / Delta_max) ** 3),
        "bound_term_max": float((hmax / Delta_max) ** 3),
        "ratio_bound_to_N_median": float((np.median(h) / Delta_max) ** 3 / N),
        "ratio_bound_to_N_max": float((hmax / Delta_max) ** 3 / N),
        "regime_valid_median": bool((np.median(h) / Delta_max) ** 3 < N),
        "regime_valid_max": bool((hmax / Delta_max) ** 3 < N),
    }


def anisotropy_report(kappas, p, n_pairs, h_frac, seed):
    """(c) grid distance vs physical distance as kappa grows.

    For each kappa we take the box [0,1] x [0,1/sqrt(kappa)] x [0,1/kappa]
    rescaled so that max_i L_i = 1, draw random pairs at physical
    L^inf-distance exactly h = h_frac, and compare
        r_grid = ||Nor(x) - Nor(y)||_inf
    with the isotropic prediction h/Delta, Delta = max_i L_i / 2^p.
    """
    rng = np.random.default_rng(seed)
    out = []
    for kap in kappas:
        L = np.array([1.0, 1.0 / np.sqrt(kap), 1.0 / kap])
        box = np.stack([np.zeros(3), L], axis=1)
        Delta = L.max() / 2 ** p
        h = h_frac * L.max()
        x = rng.random((n_pairs, 3)) * L * 0.8 + 0.1 * L
        direction = rng.normal(size=(n_pairs, 3))
        direction /= np.abs(direction).max(axis=1, keepdims=True)  # L-inf = 1
        y = x + h * direction
        y = np.clip(y, 0, L)
        gx = normalise(x, box, p)
        gy = normalise(y, box, p)
        r = np.abs(gx - gy).max(axis=1)
        out.append({
            "kappa": float(kap),
            "L": L.tolist(),
            "h": float(h),
            "Delta_from_Lmax": float(Delta),
            "isotropic_prediction_h_over_Delta": float(h / Delta),
            "r_grid_mean": float(r.mean()),
            "r_grid_max": float(r.max()),
            "measured_over_isotropic_mean": float(r.mean() / (h / Delta)),
            "measured_over_isotropic_max": float(r.max() / (h / Delta)),
            "kappa_prediction": float(kap),
        })
    return out


def roundtrip_report(points, box, p):
    """(d) encode/decode round trip on the adaptive mesh vertices."""
    g = normalise(points, box, p)
    h = hilbert_encode(g, p)
    g2 = hilbert_decode(h, p, 3)
    exact = bool((g == g2).all())
    L = box[:, 1] - box[:, 0]
    centres = box[:, 0] + (g + 0.5) * L / 2 ** p
    err = np.abs(points - centres).max(axis=1)
    bound = 0.5 * (L / 2 ** p).max()
    _, counts = np.unique(h, return_counts=True)
    return {
        "p": int(p),
        "N": int(points.shape[0]),
        "grid_roundtrip_exact": exact,
        "index_collisions": int(points.shape[0] - counts.size),
        "max_quantisation_error_mm": float(err.max()),
        "theoretical_bound_mm": float(bound),
        "bound_respected": bool(err.max() <= bound + 1e-9),
    }


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--remesh-steps", type=int, default=5)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    run = model.DiffusionRun(vol, n_target=args.n_target, seed=args.seed,
                             tau=args.tau)

    orders = list(range(3, 13))
    history = []
    for level in range(args.remesh_steps + 1):
        M, K, K0, g, gradnorm = run.operators()
        pts, box = run.mesh.points, run.mesh.box
        edges = run.mesh.edges()
        rec = {
            "level": level,
            "N": run.mesh.n_vertices,
            "n_tets": run.mesh.n_tets,
            "n_edges": int(edges.shape[0]),
            "g_min": float(g.min()), "g_max": float(g.max()),
            "collisions": collision_report(pts, box, orders),
            "hdelta": [hdelta_report(pts, box, edges, p) for p in orders],
            "roundtrip": [roundtrip_report(pts, box, p) for p in (6, 8, 10, 12)],
        }
        # smallest order with zero collisions
        zero = [c["p"] for c in rec["collisions"] if c["colliding_nodes"] == 0]
        rec["p_min_no_collision"] = int(min(zero)) if zero else None
        naive = [c["p"] for c in rec["collisions"] if c["N_le_n_d"]]
        rec["p_min_naive_condition"] = int(min(naive)) if naive else None
        history.append(rec)

        if level < args.remesh_steps:
            from phve import fem
            run.u, _ = fem.step(M, K, run.tau, run.u)
            run.remesh(gradnorm)

    box_run = run.mesh.box
    L = box_run[:, 1] - box_run[:, 0]
    out = {
        "experiment": "exp02_hypotheses",
        "params": vars(args),
        "hardware": hardware_record(),
        "domain": {"box_mm": box_run.tolist(),
                   "L_mm": L.tolist(),
                   "kappa": float(L.max() / L.min())},
        "history": history,
        "anisotropy": anisotropy_report(
            kappas=[1, 2, 3, 5, 8, 13, 20], p=8, n_pairs=20000,
            h_frac=0.01, seed=args.seed),
    }
    save_json(out, "exp02_hypotheses.json")

    print("\n(a) collisions, finest mesh (N=%d)" % history[-1]["N"])
    for c in history[-1]["collisions"]:
        print("   p=%2d  N<=n^d:%-5s  collided=%5d (%.2f%%)  max/cell=%d"
              % (c["p"], c["N_le_n_d"], c["colliding_nodes"],
                 100 * c["collision_rate"], c["max_nodes_per_cell"]))
    print("   naive condition holds from p=%s ; zero collisions only from p=%s"
          % (history[-1]["p_min_naive_condition"],
             history[-1]["p_min_no_collision"]))

    print("\n(b) h/Delta along refinement (p=8)")
    for r in history:
        hd = [x for x in r["hdelta"] if x["p"] == 8][0]
        print("   level %d  N=%6d  h_med/D=%7.1f  h_max/D=%8.1f  "
              "(h_max/D)^3/N=%10.3g  regime_ok(max)=%s"
              % (r["level"], r["N"], hd["h_over_Delta_median"],
                 hd["h_over_Delta_max"], hd["ratio_bound_to_N_max"],
                 hd["regime_valid_max"]))

    print("\n(c) anisotropy")
    for a in out["anisotropy"]:
        print("   kappa=%5.1f  r_grid_max/(h/Delta)=%6.2f  mean=%6.2f"
              % (a["kappa"], a["measured_over_isotropic_max"],
                 a["measured_over_isotropic_mean"]))

    print("\n(d) round trip, finest mesh")
    for r in history[-1]["roundtrip"]:
        print("   p=%2d  grid exact=%s  collisions=%5d  err=%.4f mm <= %.4f mm : %s"
              % (r["p"], r["grid_roundtrip_exact"], r["index_collisions"],
                 r["max_quantisation_error_mm"], r["theoretical_bound_mm"],
                 r["bound_respected"]))


if __name__ == "__main__":
    main()
