#!/usr/bin/env python3
"""
EXPERIMENT 16 -- The comparative protocol on an anisotropic domain
(closes U3, and gives R1 its first test against an actual PDE).

Every finite-element measurement so far is on the MNI152 brain domain,
whose bounding box has aspect ratio kappa = 1.24.  The anisotropy
correction R1 and the per-axis order rule are about kappa, and were tested
only *geometrically* in `exp05`: points sampled in a box, no operator, no
solve.  `UNVERIFIED.md` U3 records that the full-body reference box
(2000 x 600 x 400 mm, kappa = 5) never carried a PDE.

Construction of the anisotropic domain
--------------------------------------
The domain is obtained by scaling the *affine* of the MNI152 volume along
one axis.  Nothing else changes: the same non-convex brain mask, the same
data, the same mesh generator, the same operator.  Only the aspect ratio of
the bounding box moves.  That is exactly the variable R1 is a statement
about, so this isolates it.  A synthetic ellipsoid would have been easier
and would have thrown away the non-convexity that makes the mesh
unstructured in the first place.

What is compared, at each kappa
-------------------------------
Four orderings:

    natural      insertion order
    rcm          reverse Cuthill-McKee (graph built and timed)
    phve_cubic   PHVE at a single cubic order p -- what the paper uses now
    phve_aniso   PHVE with per-axis orders from the R1 rule
                 `choose_orders`, at the same total bit budget sum_i p_i,
                 so the code length is identical and the comparison is fair

and for each: the index-gap functionals, IC(0) and Jacobi iteration counts,
simulated cache misses at three geometries, and the ordering cost.

The prediction under test is R1: as kappa grows, phve_aniso should improve
on phve_cubic in the *mean* gap, by a factor increasing with kappa, while
the *maximum* gap should barely move (the seam theorem forbids improving
it).  exp05 measured x1.16, x1.41, x2.31 at kappa = 2, 5, 20 on point
clouds; the question here is whether that survives on a real operator and
whether it reaches the solver.

Output: results/exp16_anisotropic_domain.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                        # noqa: E402
from phve.hilbert import choose_orders                             # noqa: E402
from phve.ic0 import ic0_factor                                    # noqa: E402
from phve.metrics import (apply_permutation, order_phve,            # noqa: E402
                          order_phve_aniso, order_rcm,
                          simulate_cache_misses)
from exp10_equivariant_precond import (CACHES, count_cg,            # noqa: E402
                                       jacobi_operator)
from common import hardware_record, save_json                       # noqa: E402


def stretched_volume(vol, factors):
    """Same data and mask, affine scaled per axis.  Only the aspect ratio
    of the physical domain changes."""
    A = vol.affine.copy()
    A[:3, :3] = np.diag(np.asarray(factors, float)) @ A[:3, :3]
    A[:3, 3] = np.asarray(factors, float) * A[:3, 3]
    return model.Volume(data=vol.data, affine=A,
                        name="%s-stretched-%s" % (vol.name, list(factors)))


def gap_stats(A):
    C = A.tocoo()
    g = np.abs(C.row.astype(np.int64) - C.col.astype(np.int64))
    return {"max": int(g.max()), "mean": float(g.mean()),
            "median": float(np.median(g))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stretch", type=float, nargs="+",
                    default=[1.0, 2.0, 4.0, 8.0],
                    help="scale factor applied to the first axis")
    ap.add_argument("--n-target", type=int, default=20000)
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-10)
    args = ap.parse_args()

    base = model.load_mni152(2)
    records = []

    for seed in args.seeds:
        for s in args.stretch:
            vol = stretched_volume(base, [s, 1.0, 1.0])
            run = model.DiffusionRun(vol, n_target=args.n_target, seed=seed,
                                     tau=args.tau)
            M_, K, K0, g, gradnorm = run.operators()
            run.u, _ = fem.step(M_, K, args.tau, run.u)
            run.remesh(gradnorm)
            M_, K, K0, g, gradnorm = run.operators()
            A = (M_ + args.tau * K).tocsr()
            b = M_ @ run.u
            n = A.shape[0]

            box = np.asarray(run.mesh.box, float)
            L = box[:, 1] - box[:, 0]
            kappa = float(L.max() / L.min())
            budget = 3 * args.order
            orders = choose_orders(box, budget)

            print("stretch=%.1f seed=%d  N=%d  L=%s  kappa=%.2f  "
                  "R1 orders=%s (budget %d)"
                  % (s, seed, n, np.round(L, 1).tolist(), kappa,
                     orders.tolist(), budget))

            perms = {}
            perms["natural"] = (np.arange(n), 0.0)
            perms["rcm"] = order_rcm(run.mesh)
            perms["phve_cubic"] = order_phve(run.mesh, args.order)
            perms["phve_aniso"] = order_phve_aniso(run.mesh, orders)

            rec = {"seed": seed, "stretch": s, "N": int(n), "nnz": int(A.nnz),
                   "L_mm": L.tolist(), "kappa": kappa,
                   "bit_budget": budget, "R1_orders": orders.tolist(),
                   "orderings": {}}

            for name, (perm, t_order) in perms.items():
                Ap = apply_permutation(A, perm).tocsr()
                bp = b[perm]
                e = {"order_time_s": t_order, "gaps": gap_stats(Ap)}

                ic = ic0_factor(Ap)
                k, ok, _ = count_cg(Ap, bp, ic.as_operator(), args.rtol, 20000)
                e["ic0_iters"] = k
                e["ic0_converged"] = ok
                e["ic0_nnz"] = ic.nnz

                k, ok, _ = count_cg(Ap, bp, jacobi_operator(Ap), args.rtol, 20000)
                e["jacobi_iters"] = k
                e["jacobi_converged"] = ok

                e["cache"] = {}
                for label, sets, ways, line in CACHES:
                    c = simulate_cache_misses(Ap, line=line, sets=sets,
                                              ways=ways)
                    e["cache"][label] = c["misses"]

                rec["orderings"][name] = e
                print("   %-11s gap mean %9.1f  median %7.1f  max %8d | "
                      "IC0 %4d  Jacobi %4d | L1 miss %9d | order %.4f s"
                      % (name, e["gaps"]["mean"], e["gaps"]["median"],
                         e["gaps"]["max"], e["ic0_iters"], e["jacobi_iters"],
                         e["cache"][CACHES[0][0]], t_order))
            records.append(rec)
            sys.stdout.flush()

    # ------------------------------------------------------------------
    print("\n=== R1 under test: does the per-axis rule beat the cubic order? ===")
    print("  kappa |  mean gap cubic -> aniso  | factor | max gap cubic -> aniso")
    rows = []
    for r in sorted(records, key=lambda x: (x["seed"], x["kappa"])):
        c = r["orderings"]["phve_cubic"]["gaps"]
        a = r["orderings"]["phve_aniso"]["gaps"]
        row = {"seed": r["seed"], "kappa": r["kappa"], "N": r["N"],
               "mean_cubic": c["mean"], "mean_aniso": a["mean"],
               "mean_factor": c["mean"] / max(a["mean"], 1e-12),
               "max_cubic": c["max"], "max_aniso": a["max"],
               "max_factor": c["max"] / max(a["max"], 1),
               "ic0_cubic": r["orderings"]["phve_cubic"]["ic0_iters"],
               "ic0_aniso": r["orderings"]["phve_aniso"]["ic0_iters"],
               "ic0_rcm": r["orderings"]["rcm"]["ic0_iters"],
               "jacobi_spread": max(r["orderings"][k]["jacobi_iters"]
                                    for k in r["orderings"])
               - min(r["orderings"][k]["jacobi_iters"]
                     for k in r["orderings"])}
        rows.append(row)
        print("  %5.2f | %10.1f -> %10.1f | x%5.2f | %8d -> %8d (x%.2f)"
              % (row["kappa"], row["mean_cubic"], row["mean_aniso"],
                 row["mean_factor"], row["max_cubic"], row["max_aniso"],
                 row["max_factor"]))

    print("\n=== Does it reach the solver? ===")
    for row in rows:
        print("  kappa %5.2f  IC(0) iters: rcm %4d  cubic %4d  aniso %4d"
              "   | Jacobi spread across all four orderings: %d"
              % (row["kappa"], row["ic0_rcm"], row["ic0_cubic"],
                 row["ic0_aniso"], row["jacobi_spread"]))

    out = {"experiment": "exp16_anisotropic_domain", "params": vars(args),
           "hardware": hardware_record(), "caches": [c[0] for c in CACHES],
           "records": records, "R1_summary": rows}
    save_json(out, "exp16_anisotropic_domain.json")


if __name__ == "__main__":
    main()
