#!/usr/bin/env python3
"""
EXPERIMENT 18 -- Numerical-analysis properties of the scheme (measures U7).

U7 states, correctly, that the scheme is used only as a *generator of
realistic sparse SPD systems on a changing unstructured mesh*, and that a
discrete maximum principle and convergence to the continuous quasi-linear
problem are **not** established.  That is a deliberate limitation, not an
oversight.  But "not established" is currently asserted rather than
measured, and the paper would be stronger if it said by how much the
condition fails and whether the failure is visible in the solution.

Three things are measured, none of which is a proof.

(A) *The non-obtuse condition.*  For P1 elements the discrete maximum
    principle requires the off-diagonal entries of the stiffness matrix to
    be non-positive, which for tetrahedra is the non-obtuse (Delaunay-like)
    angle condition.  Our adaptive Delaunay meshes do not enforce it.  The
    fraction of positive off-diagonal entries, and their weight relative to
    the negative ones, say exactly how far the mesh is from the condition.

(B) *Whether it actually bites.*  A necessary consequence of the discrete
    maximum principle is that no time step creates a new extremum:
    min(u^n) <= u^{n+1} <= max(u^n) entrywise.  This is checked at every
    step, and the size of any overshoot is recorded.  A mesh can violate
    (A) and still never violate (B) in practice; conversely a violation of
    (B) is a hard fact, not a technicality.

(C) *Observed convergence order in space.*  Self-convergence: solve on a
    sequence of refined meshes and compare against the finest as reference,
    interpolating by nearest neighbour on the common vertex set.  This
    estimates an observed rate.  It is not a convergence proof and cannot
    be one -- the continuous problem's solution is not available -- but a
    rate near the expected value is evidence the discretisation is sane,
    and a rate near zero would be evidence it is not.

Output: results/exp18_scheme_properties.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                        # noqa: E402
from common import hardware_record, save_json                      # noqa: E402


def offdiag_positivity(K):
    """Fraction and weight of positive off-diagonal stiffness entries."""
    C = K.tocoo()
    off = C.row != C.col
    vals = C.data[off]
    pos = vals > 0
    return {"n_offdiag": int(vals.size),
            "n_positive": int(pos.sum()),
            "fraction_positive": float(pos.sum() / max(vals.size, 1)),
            "sum_positive": float(vals[pos].sum()) if pos.any() else 0.0,
            "sum_negative_abs": float(np.abs(vals[~pos]).sum()),
            "weight_ratio": float(vals[pos].sum()
                                  / max(np.abs(vals[~pos]).sum(), 1e-300))
            if pos.any() else 0.0,
            "max_positive": float(vals[pos].max()) if pos.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--n-target", type=int, default=20000)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--refine-fraction", type=float, default=0.03,
                    help="fraction of elements refined per remesh; the "
                         "DiffusionRun default of 0.15 inserts ~0.9 N new "
                         "vertices per remesh and makes the mesh grow "
                         "exponentially, which is unusable over 10 steps")
    ap.add_argument("--conv-sizes", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000])
    args = ap.parse_args()

    vol = model.load_mni152(2)
    out = {"experiment": "exp18_scheme_properties", "params": vars(args),
           "hardware": hardware_record()}

    # ------------------------------------------------------------------
    # (A) and (B)
    # ------------------------------------------------------------------
    print("=== (A) non-obtuse condition and (B) discrete maximum principle ===")
    runs = []
    for seed in args.seeds:
        run = model.DiffusionRun(vol, n_target=args.n_target, seed=seed,
                                 tau=args.tau,
                                 refine_fraction=args.refine_fraction)
        steps = []
        worst_over = 0.0
        worst_under = 0.0
        n_violating = 0
        for it in range(args.steps):
            M, K, K0, g, gradnorm = run.operators()
            pos = offdiag_positivity(K)
            u_old = run.u.copy()
            run.u, _ = fem.step(M, K, args.tau, run.u)
            lo, hi = float(u_old.min()), float(u_old.max())
            over = float(max(0.0, run.u.max() - hi))
            under = float(max(0.0, lo - run.u.min()))
            rng_u = max(hi - lo, 1e-300)
            viol = (over > 1e-12 * rng_u) or (under > 1e-12 * rng_u)
            n_violating += int(viol)
            worst_over = max(worst_over, over / rng_u)
            worst_under = max(worst_under, under / rng_u)
            steps.append({"step": it, "N": int(run.mesh.n_vertices),
                          "offdiag": pos,
                          "overshoot_rel": over / rng_u,
                          "undershoot_rel": under / rng_u,
                          "dmp_violated": bool(viol)})
            if True:
                print("   seed=%d step=%2d N=%6d | positive off-diagonals"
                      " %6.2f %% (weight %.3f) | overshoot %.2e undershoot %.2e"
                      % (seed, it, run.mesh.n_vertices,
                         100 * pos["fraction_positive"], pos["weight_ratio"],
                         over / rng_u, under / rng_u))
            run.remesh(gradnorm)
            sys.stdout.flush()
        runs.append({"seed": seed, "steps": steps,
                     "n_steps_violating_dmp": n_violating,
                     "worst_overshoot_rel": worst_over,
                     "worst_undershoot_rel": worst_under})
        print("   seed=%d: DMP violated on %d of %d steps; worst overshoot"
              " %.3e, worst undershoot %.3e (relative to the range of u)"
              % (seed, n_violating, args.steps, worst_over, worst_under))
    out["maximum_principle"] = runs

    fr = [s["offdiag"]["fraction_positive"] for r in runs for s in r["steps"]]
    print("   positive off-diagonal fraction over all steps: min %.4f max %.4f"
          % (min(fr), max(fr)))
    out["offdiag_positive_fraction_range"] = [min(fr), max(fr)]

    # ------------------------------------------------------------------
    # (C) self-convergence in space
    # ------------------------------------------------------------------
    print("\n=== (C) observed self-convergence in space ===")
    from scipy.spatial import cKDTree
    sols = []
    for n_target in args.conv_sizes:
        run = model.DiffusionRun(vol, n_target=n_target, seed=1, tau=args.tau)
        for _ in range(4):
            M, K, K0, g, gradnorm = run.operators()
            run.u, _ = fem.step(M, K, args.tau, run.u)
        h = float(np.median(run.mesh.edge_lengths()))
        sols.append({"n_target": n_target, "N": int(run.mesh.n_vertices),
                     "h_median": h, "pts": run.mesh.points.copy(),
                     "u": run.u.copy()})
        print("   built N=%6d  median edge %.3f mm" % (sols[-1]["N"], h))
        sys.stdout.flush()

    ref = sols[-1]
    tree = cKDTree(ref["pts"])
    rows = []
    for s in sols[:-1]:
        _, nn = tree.query(s["pts"], k=1)
        err = s["u"] - ref["u"][nn]
        rows.append({"N": s["N"], "h_median": s["h_median"],
                     "l2_rel": float(np.linalg.norm(err)
                                     / max(np.linalg.norm(ref["u"][nn]), 1e-300)),
                     "linf": float(np.abs(err).max())})
        print("   N=%6d h=%.3f  rel L2 vs reference %.4e   Linf %.4e"
              % (rows[-1]["N"], rows[-1]["h_median"], rows[-1]["l2_rel"],
                 rows[-1]["linf"]))
    if len(rows) >= 2:
        hs = np.log([r["h_median"] for r in rows])
        es = np.log([r["l2_rel"] for r in rows])
        slope = float(np.polyfit(hs, es, 1)[0])
        print("   observed order in the median edge length: %.2f" % slope)
        out["self_convergence"] = {"rows": rows, "observed_order": slope,
                                   "reference_N": ref["N"]}

    save_json(out, "exp18_scheme_properties.json")


if __name__ == "__main__":
    main()
