#!/usr/bin/env python3
"""
EXPERIMENT 20 -- Which hypotheses do our meshes actually satisfy?

Purpose
-------
Before attempting the master theorem, establish which of its hypotheses
survive on the meshes the paper actually uses.  Proving a theorem from
hypotheses the experiments violate would be worse than proving nothing.

The quantity that decides everything
------------------------------------
Reconstructing the dyadic counting argument: for an edge whose endpoints
are at distance <= h, the number of edges separated at level l is at most
D c_2 S N h 2^l, and two points sharing a level-l cell have index gap at
most c_1 N 2^{-dl}.  Requiring |i-j| > t forces 2^{dl} < c_1 N / t, so

    P(|i-j| > t)  <~  (D c_2 S N h / |E|) (c_1 N)^{1/d} t^{-1/d}
                   ~   2 c_2 S c_1^{1/d} ( h N^{1/d} ) t^{-1/d}.

So the bound is independent of N **exactly when h N^{1/d} = O(1)**, which
on a domain of fixed volume is quasi-uniformity.  It is not a technical
convenience in the hypothesis list: it is the whole mechanism.

Our meshes are produced by adaptive refinement, i.e. deliberately graded,
which is the negation of quasi-uniformity.  This script measures:

 (1) h_max N^{1/d} and h_med N^{1/d} along the mesh sequence -- the
     quantity that must stay bounded;
 (2) the grading ratio h_max/h_min and its behaviour under refinement;
 (3) shape regularity (inradius/circumradius) of the tetrahedra, which the
     mesh-size argument silently assumes;
 (4) for comparison, the same quantities on the *initial* (unrefined)
     meshes, which are much closer to quasi-uniform.

The point is to find out whether the observed N-independence of the gap
distribution (exp08) happens *despite* the hypothesis failing -- in which
case the theorem needs a different mechanism, not a better proof.

Output: results/exp20_mesh_regularity_audit.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                        # noqa: E402
from common import hardware_record, save_json                      # noqa: E402


def shape_regularity(mesh):
    """rho = inradius/circumradius per tetrahedron, normalised so that the
    regular tetrahedron gives 1.  Reported by percentiles."""
    p = mesh.points[mesh.tets]
    a, b, c, d = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    vol = np.abs(np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a))) / 6.0

    def area(u, v, w):
        return 0.5 * np.linalg.norm(np.cross(v - u, w - u), axis=1)

    S = area(a, b, c) + area(a, b, d) + area(a, c, d) + area(b, c, d)
    r_in = 3.0 * vol / np.maximum(S, 1e-300)

    # circumradius via the standard determinant formula
    A = b - a
    B = c - a
    C = d - a
    na, nb, nc = (A ** 2).sum(1), (B ** 2).sum(1), (C ** 2).sum(1)
    num = np.linalg.norm(na[:, None] * np.cross(B, C)
                         + nb[:, None] * np.cross(C, A)
                         + nc[:, None] * np.cross(A, B), axis=1)
    r_out = num / np.maximum(12.0 * vol, 1e-300)
    rho = 3.0 * r_in / np.maximum(r_out, 1e-300)   # regular tet -> 1
    return rho, vol


def mesh_record(mesh, d=3):
    el = mesh.edge_lengths()
    N = mesh.n_vertices
    rho, vol = shape_regularity(mesh)
    box = np.asarray(mesh.box, float)
    L = box[:, 1] - box[:, 0]
    return {
        "N": int(N), "n_tets": int(mesh.n_tets),
        "h_max": float(el.max()), "h_min": float(el.min()),
        "h_median": float(np.median(el)), "h_mean": float(el.mean()),
        "h_p99": float(np.percentile(el, 99)),
        "grading_ratio": float(el.max() / max(el.min(), 1e-300)),
        # the quantity that must be O(1) for the bound to be N-independent
        "h_max_N_pow": float(el.max() * N ** (1.0 / d)),
        "h_med_N_pow": float(np.median(el) * N ** (1.0 / d)),
        "h_p99_N_pow": float(np.percentile(el, 99) * N ** (1.0 / d)),
        "shape_rho_min": float(rho.min()),
        "shape_rho_p1": float(np.percentile(rho, 1)),
        "shape_rho_median": float(np.median(rho)),
        "vol_ratio_max_min": float(vol.max() / max(vol.min(), 1e-300)),
        "L_mm": L.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--remesh-steps", type=int, default=1,
                    help="number of adaptive remeshes applied, as in exp10")
    ap.add_argument("--refine-fraction", type=float, default=0.15)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    initial, refined = [], []

    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed,
                                     tau=args.tau,
                                     refine_fraction=args.refine_fraction)
            r0 = mesh_record(run.mesh)
            r0.update({"seed": seed, "n_target": n_target, "stage": "initial"})
            initial.append(r0)

            for _ in range(args.remesh_steps):
                M, K, K0, g, gradnorm = run.operators()
                run.u, _ = fem.step(M, K, args.tau, run.u)
                run.remesh(gradnorm)
            r1 = mesh_record(run.mesh)
            r1.update({"seed": seed, "n_target": n_target, "stage": "refined"})
            refined.append(r1)

            print("seed=%d target=%6d | initial N=%6d h_max*N^(1/3)=%7.2f"
                  " grading=%7.1f | refined N=%6d h_max*N^(1/3)=%7.2f"
                  " grading=%8.1f"
                  % (seed, n_target, r0["N"], r0["h_max_N_pow"],
                     r0["grading_ratio"], r1["N"], r1["h_max_N_pow"],
                     r1["grading_ratio"]))
            sys.stdout.flush()

    # ------------------------------------------------------------------
    print("\n=== (1) The quantity that must be O(1): h * N^(1/d) ===")
    print("  stage    |    N   | h_max*N^(1/3) | h_p99*N^(1/3) | h_med*N^(1/3)")
    for label, rows in (("initial", initial), ("refined", refined)):
        for r in sorted(rows, key=lambda x: (x["seed"], x["N"])):
            if r["seed"] != args.seeds[0]:
                continue
            print("  %-8s | %6d | %13.2f | %13.2f | %13.2f"
                  % (label, r["N"], r["h_max_N_pow"], r["h_p99_N_pow"],
                     r["h_med_N_pow"]))

    def trend(rows, key):
        rs = [r for r in rows if r["seed"] == args.seeds[0]]
        rs.sort(key=lambda x: x["N"])
        if len(rs) < 2:
            return None
        x = np.log([r["N"] for r in rs])
        y = np.log([max(r[key], 1e-300) for r in rs])
        return float(np.polyfit(x, y, 1)[0])

    print("\n=== (2) Trends in N (slope of log-log; 0 means bounded) ===")
    out_tr = {}
    for label, rows in (("initial", initial), ("refined", refined)):
        for key in ("h_max_N_pow", "h_p99_N_pow", "h_med_N_pow",
                    "grading_ratio"):
            s = trend(rows, key)
            out_tr["%s_%s" % (label, key)] = s
            print("   %-8s %-16s slope %+.3f" % (label, key, s))

    print("\n=== (3) Shape regularity (1 = regular tetrahedron) ===")
    for label, rows in (("initial", initial), ("refined", refined)):
        mn = min(r["shape_rho_min"] for r in rows)
        p1 = min(r["shape_rho_p1"] for r in rows)
        md = np.median([r["shape_rho_median"] for r in rows])
        print("   %-8s rho: min %.2e   1st pct %.4f   median %.4f"
              % (label, mn, p1, md))

    print("\n=== Verdict ===")
    s_ref = out_tr["refined_h_max_N_pow"]
    s_med = out_tr["refined_h_med_N_pow"]
    print("   quasi-uniformity (h_max N^(1/d) bounded) on refined meshes: %s"
          % ("plausible" if abs(s_ref) < 0.05 else
             "FAILS, slope %+.3f" % s_ref))
    print("   the same for the median edge: %s"
          % ("plausible" if abs(s_med) < 0.05 else
             "fails, slope %+.3f" % s_med))

    out = {"experiment": "exp20_mesh_regularity_audit", "params": vars(args),
           "hardware": hardware_record(), "initial": initial,
           "refined": refined, "trends": out_tr}
    save_json(out, "exp20_mesh_regularity_audit.json")


if __name__ == "__main__":
    main()
