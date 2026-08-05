#!/usr/bin/env python3
"""
EXPERIMENT 10 -- Permutation-equivariant preconditioners: the claimed
positive result, established from scratch.

Why this script exists
----------------------
The brainstorming note `DIRECTION.md` (Sec. 2.5) asserted that with a
*permutation-equivariant* preconditioner the iteration count is invariant
under renumbering, so that the only remaining differences are memory
traffic and the cost of computing the ordering -- and that above a cache
threshold PHVE therefore strictly dominates reverse Cuthill-McKee.  No
script in this repository produced those numbers.  Under the working rule
that no figure enters the paper without a versioned, seeded script, the
claim had to be treated as unestablished.  This file establishes it, or
refutes it.

What is measured
----------------
For each mesh size N and each ordering (natural / RCM / PHVE):

  * the PCG iteration count for four preconditioners

        none      -- control; equivariant, iteration count must be invariant
        jacobi    -- M^{-1} = D^{-1};                   equivariant
        cheb(k)   -- M^{-1} = q_k(D^{-1/2} A D^{-1/2}); equivariant
        ic0       -- incomplete Cholesky, fixed pattern; NOT equivariant

    Equivariance is exact in exact arithmetic: for a permutation P,
    q(P A P^T) = P q(A) P^T for any function q defined by a polynomial in
    A and D.  Any observed spread is floating-point summation order, and is
    bounded here by the spread of the unpreconditioned control.

  * the simulated cache misses of one CSR sparse matrix-vector product at
    three cache geometries (L1 32 KiB, L2 1 MiB, L3 16 MiB), and the
    *total* misses over the whole solve, iterations x misses-per-SpMV.
    The latter is the figure of merit: with an equivariant preconditioner
    the iteration count cancels, so total traffic is decided by locality
    alone.

  * the wall-clock cost of computing the ordering itself, which is the
    other term that does not cancel.

The sweep deliberately runs past the point where the working set (8N bytes
for the solution vector) exceeds each cache level, because the whole
question is whether the locality advantage of a space-filling curve only
appears once the band no longer fits in cache.

Output: results/exp10_equivariant_precond.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                       # noqa: E402
from phve.ic0 import ic0_factor                                   # noqa: E402
from phve.metrics import (Timer, apply_permutation, order_phve,    # noqa: E402
                          order_rcm, simulate_cache_misses)
from common import hardware_record, save_json                      # noqa: E402

SCHEMES = ("natural", "rcm", "phve")

# (label, sets, ways, line): total size = sets * ways * line
CACHES = [
    ("L1 32 KiB 8-way", 64, 8, 64),
    ("L2 1 MiB 16-way", 1024, 16, 64),
    ("L3 16 MiB 16-way", 16384, 16, 64),
]


# ----------------------------------------------------------------------
# Equivariant preconditioners
# ----------------------------------------------------------------------

def jacobi_operator(A):
    """M^{-1} = D^{-1}.  Commutes with permutation exactly."""
    dinv = 1.0 / A.diagonal()
    return spla.LinearOperator(A.shape, matvec=lambda x: dinv * x,
                               dtype=np.float64)


def _scaled_matrix(A):
    """Return (D^{-1/2} A D^{-1/2}, d^{-1/2}) for the symmetric scaling."""
    dm = 1.0 / np.sqrt(A.diagonal())
    S = sp.diags(dm) @ A @ sp.diags(dm)
    return S.tocsr(), dm


def _spectral_bound(S, iters=30, seed=0):
    """Upper bound on lambda_max(S) by a fixed number of Lanczos-free power
    iterations, then inflated by 5 % for safety.  Deterministic given the
    seed; the same starting vector is used for every ordering by permuting
    it, so the bound is itself equivariant."""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(S.shape[0])
    x /= np.linalg.norm(x)
    lam = 0.0
    for _ in range(iters):
        y = S @ x
        lam = float(np.linalg.norm(y))
        if lam == 0.0:
            break
        x = y / lam
    return 1.05 * lam


def chebyshev_operator(A, degree, lmax, ratio=30.0):
    """M^{-1} = q_k(S) in the symmetrically scaled variable, where q_k is
    the degree-`degree` Chebyshev approximation to the inverse on the
    interval [lmax/ratio, lmax].

    Applied as x -> D^{-1/2} q_k(S) D^{-1/2} x.  Every operation is a
    polynomial in A and D, hence exactly permutation-equivariant.
    """
    S, dm = _scaled_matrix(A)
    a = lmax / ratio
    b = lmax
    theta = 0.5 * (b + a)
    delta = 0.5 * (b - a)

    def apply(x):
        r = dm * x
        # Chebyshev semi-iteration for S z = r, starting from z = 0
        z = np.zeros_like(r)
        p = np.zeros_like(r)
        alpha = beta = 0.0
        for k in range(degree + 1):
            res = r - S @ z
            if k == 0:
                alpha = 1.0 / theta
                p = res.copy()
            elif k == 1:
                beta = 0.5 * (delta * alpha) ** 2
                alpha = 1.0 / (theta - beta / alpha)
                p = res + beta * p
            else:
                beta = (delta * alpha / 2.0) ** 2
                alpha = 1.0 / (theta - beta / alpha)
                p = res + beta * p
            z = z + alpha * p
        return dm * z

    return spla.LinearOperator(A.shape, matvec=apply, dtype=np.float64)


def count_cg(A, b, M, rtol, maxiter):
    """PCG iteration count and convergence flag."""
    n = {"k": 0}
    x, info = spla.cg(A, b, rtol=rtol, atol=0.0, maxiter=maxiter, M=M,
                      callback=lambda z: n.__setitem__("k", n["k"] + 1))
    return n["k"], info == 0, x


# ----------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[4000, 8000, 16000, 32000, 64000, 128000])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--maxiter", type=int, default=20000)
    ap.add_argument("--cheb-degree", type=int, default=4)
    ap.add_argument("--power-iters", type=int, default=30)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    records = []

    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed,
                                     tau=args.tau)
            M_, K, K0, g, gradnorm = run.operators()
            run.u, _ = fem.step(M_, K, args.tau, run.u)
            run.remesh(gradnorm)
            M_, K, K0, g, gradnorm = run.operators()
            A = (M_ + args.tau * K).tocsr()
            b = M_ @ run.u
            n = A.shape[0]

            # --- orderings, with the honest cost of computing each --------
            perms = {}
            perms["natural"] = (np.arange(n), 0.0)
            with Timer() as tm:
                pass
            p_rcm, t_rcm = order_rcm(run.mesh)
            perms["rcm"] = (p_rcm, t_rcm)
            p_phve, t_phve = order_phve(run.mesh, args.order)
            perms["phve"] = (p_phve, t_phve)

            # The spectral bound must be the SAME number for every ordering,
            # otherwise the Chebyshev polynomials differ and equivariance is
            # broken by construction rather than by arithmetic.  Compute it
            # once, on the natural ordering.
            S_nat, _ = _scaled_matrix(A)
            lmax = _spectral_bound(S_nat, iters=args.power_iters, seed=seed)

            rec = {"seed": seed, "N": int(n), "nnz": int(A.nnz),
                   "working_set_kib": 8.0 * n / 1024.0,
                   "lmax_bound": lmax, "orderings": {}}

            for name in SCHEMES:
                perm, t_order = perms[name]
                Ap = apply_permutation(A, perm).tocsr()
                bp = b[perm]

                entry = {"order_time_s": t_order}

                # index-gap functionals, for cross-reference with exp03
                Ac = Ap.tocoo()
                gaps = np.abs(Ac.row.astype(np.int64) - Ac.col.astype(np.int64))
                entry["gap_max"] = int(gaps.max())
                entry["gap_mean"] = float(gaps.mean())
                entry["gap_median"] = float(np.median(gaps))

                # --- cache behaviour of one SpMV -------------------------
                cache = {}
                for label, sets, ways, line in CACHES:
                    c = simulate_cache_misses(Ap, line=line, sets=sets,
                                              ways=ways)
                    cache[label] = {"misses": c["misses"],
                                    "accesses": c["accesses"],
                                    "miss_rate": c["miss_rate"]}
                entry["cache"] = cache

                # --- preconditioners -------------------------------------
                its = {}

                k, ok, _ = count_cg(Ap, bp, None, args.rtol, args.maxiter)
                its["none"] = {"iters": k, "converged": ok}

                k, ok, _ = count_cg(Ap, bp, jacobi_operator(Ap), args.rtol,
                                    args.maxiter)
                its["jacobi"] = {"iters": k, "converged": ok}

                cheb = chebyshev_operator(Ap, args.cheb_degree, lmax)
                k, ok, _ = count_cg(Ap, bp, cheb, args.rtol, args.maxiter)
                its["cheb%d" % args.cheb_degree] = {"iters": k,
                                                    "converged": ok}

                ic = ic0_factor(Ap)
                k, ok, _ = count_cg(Ap, bp, ic.as_operator(), args.rtol,
                                    args.maxiter)
                its["ic0"] = {"iters": k, "converged": ok,
                              "nnz": ic.nnz, "shift": ic.shift}

                entry["iters"] = its

                # --- total traffic: iterations x misses per SpMV ----------
                total = {}
                for pc, v in its.items():
                    total[pc] = {label: v["iters"] * cache[label]["misses"]
                                 for label, *_ in CACHES}
                entry["total_misses"] = total

                rec["orderings"][name] = entry

            records.append(rec)

            eq = rec["orderings"]
            print("seed=%d N=%6d ws=%7.1f KiB" % (seed, n,
                                                  rec["working_set_kib"]))
            for pc in ("none", "jacobi", "cheb%d" % args.cheb_degree, "ic0"):
                print("   %-8s its: nat %5d  rcm %5d  phve %5d   spread %.2f%%"
                      % (pc,
                         eq["natural"]["iters"][pc]["iters"],
                         eq["rcm"]["iters"][pc]["iters"],
                         eq["phve"]["iters"][pc]["iters"],
                         100.0 * (max(eq[s]["iters"][pc]["iters"] for s in SCHEMES)
                                  - min(eq[s]["iters"][pc]["iters"] for s in SCHEMES))
                         / max(1, max(eq[s]["iters"][pc]["iters"] for s in SCHEMES))))
            for label, *_ in CACHES:
                print("   %-16s misses/SpMV: rcm %10d  phve %10d  (phve/rcm %.3f)"
                      % (label, eq["rcm"]["cache"][label]["misses"],
                         eq["phve"]["cache"][label]["misses"],
                         eq["phve"]["cache"][label]["misses"]
                         / max(eq["rcm"]["cache"][label]["misses"], 1)))
            print("   order time: rcm %.4f s  phve %.4f s  (rcm/phve %.2f)"
                  % (eq["rcm"]["order_time_s"], eq["phve"]["order_time_s"],
                     eq["rcm"]["order_time_s"]
                     / max(eq["phve"]["order_time_s"], 1e-12)))
            sys.stdout.flush()

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    pcs = ["none", "jacobi", "cheb%d" % args.cheb_degree, "ic0"]
    summary = {"equivariance": {}, "cache_crossover": {}, "order_cost": []}

    print("\n=== Equivariance: relative spread of the iteration count ===")
    for pc in pcs:
        rows = []
        for r in records:
            v = [r["orderings"][s]["iters"][pc]["iters"] for s in SCHEMES]
            rows.append({"N": r["N"], "seed": r["seed"],
                         "natural": v[0], "rcm": v[1], "phve": v[2],
                         "spread_rel": (max(v) - min(v)) / max(1, max(v))})
        worst = max(rows, key=lambda x: x["spread_rel"])
        summary["equivariance"][pc] = {"rows": rows,
                                       "worst_spread_rel": worst["spread_rel"],
                                       "worst_at_N": worst["N"]}
        print("  %-8s worst spread %.3f %% at N=%d"
              % (pc, 100 * worst["spread_rel"], worst["N"]))

    print("\n=== Cache crossover (PHVE/RCM misses per SpMV) ===")
    for label, *_ in CACHES:
        rows = [{"N": r["N"], "seed": r["seed"],
                 "ratio": r["orderings"]["phve"]["cache"][label]["misses"]
                 / max(r["orderings"]["rcm"]["cache"][label]["misses"], 1)}
                for r in records]
        Ns = sorted({row["N"] for row in rows})
        crossover = None
        for N0 in Ns:
            above = [row for row in rows if row["N"] >= N0]
            if above and all(row["ratio"] < 1.0 for row in above):
                crossover = N0
                break
        summary["cache_crossover"][label] = {"rows": rows,
                                             "crossover_N": crossover}
        print("  %-16s crossover at N >= %s" % (label, crossover))
        for row in sorted(rows, key=lambda x: (x["N"], x["seed"])):
            print("      N=%6d seed=%d ratio %.3f"
                  % (row["N"], row["seed"], row["ratio"]))

    print("\n=== Ordering cost ===")
    for r in records:
        e = r["orderings"]
        row = {"N": r["N"], "seed": r["seed"],
               "rcm_s": e["rcm"]["order_time_s"],
               "phve_s": e["phve"]["order_time_s"],
               "speedup": e["rcm"]["order_time_s"]
               / max(e["phve"]["order_time_s"], 1e-12)}
        summary["order_cost"].append(row)
        print("  N=%6d seed=%d  rcm %.4f s  phve %.4f s  x%.2f"
              % (row["N"], row["seed"], row["rcm_s"], row["phve_s"],
                 row["speedup"]))

    out = {"experiment": "exp10_equivariant_precond",
           "params": vars(args), "hardware": hardware_record(),
           "caches": [c[0] for c in CACHES], "preconditioners": pcs,
           "records": records, "summary": summary}
    save_json(out, "exp10_equivariant_precond.json")


if __name__ == "__main__":
    main()
