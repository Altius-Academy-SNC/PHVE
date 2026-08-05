#!/usr/bin/env python3
"""
EXPERIMENT 04 -- The comparative protocol (Step 3 of the work plan).

The model problem is integrated in time with adaptive remeshing every m
steps.  At every step and for each of the three orderings we measure

    bandwidth (max and mean),  ILU fill,  PCG iteration count,
    simulated cache misses of one SpMV,  and the wall-clock times of
    renumbering / factorisation / solution.

The decisive figure is the *total* time including renumbering, as m varies.
Renumbering is performed at every remeshing; between two remeshings the
numbering is reused, and the nodes inserted by the refinement keep their
insertion labels, which is what a real adaptive code does.

The trajectory itself is driven once, in the natural ordering; by the
Equivalence Lemma (verified in exp01) the iterates are the same for every
ordering up to round-off, so this does not bias the comparison.

Output: results/exp04_comparative.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                      # noqa: E402
from phve.ic0 import ic0_factor                               # noqa: E402
from phve.metrics import (Timer, apply_permutation, ilu_fill,    # noqa: E402
                          order_phve, order_rcm,
                          simulate_cache_misses)
from common import hardware_record, save_json                    # noqa: E402

SCHEMES = ("natural", "rcm", "phve")


def gap_stats(A, perm):
    n = A.shape[0]
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    C = A.tocoo()
    d = np.abs(inv[C.row.astype(np.int64)] - inv[C.col.astype(np.int64)])
    return {"max": int(d.max()), "mean": float(d.mean()),
            "median": float(np.median(d)), "p99": float(np.percentile(d, 99))}


def _ilu(Ap, drop_tol, fill_factor=10.0):
    """Incomplete LU respecting the current ordering.

    ``permc_spec="NATURAL"`` is essential: with the SuperLU default
    (COLAMD) the factorisation silently reorders the matrix itself, and
    the experiment would measure COLAMD rather than the ordering under
    test.  ``SymmetricMode`` and a zero pivot threshold keep the factor as
    close to a Cholesky factor as SuperLU allows, which is what a
    conjugate-gradient iteration needs.
    """
    return spla.spilu(Ap.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor,
                      permc_spec="NATURAL", diag_pivot_thresh=0.0,
                      options=dict(SymmetricMode=True))


def _cg(Ap, bp, rtol, maxiter, prec=None):
    it = {"n": 0}
    with Timer() as ts:
        x, info = spla.cg(Ap, bp, rtol=rtol, atol=0.0, maxiter=maxiter, M=prec,
                          callback=lambda xk: it.__setitem__("n", it["n"] + 1))
    return x, info, it["n"], ts.t


def measure(A, b, perm, t_reorder, drop_tol, rtol, maxiter=2000,
            n_apply=20, with_ilu=True):
    """Cost of one implicit step under a given ordering.

    Three solvers are timed.

    * Unpreconditioned CG is permutation-invariant (Equivalence Lemma), so
      its iteration count is a control: it must be the same for the three
      orderings.
    * IC(0) keeps the sparsity pattern of tril(A) fixed, so its fill is
      identical for every ordering *by construction*.  A numbering can
      therefore no longer be punished by extra fill; the only things that
      can differ are the quality of the preconditioner (iteration count)
      and the speed of the triangular solves (cache behaviour).  This is
      the honest comparison, and it is the headline one.
    * Threshold ILU chooses its own fill and is kept only for comparison
      with the previous version of this study.
    """
    with Timer() as tp:
        Ap = apply_permutation(A, perm).tocsr()
    bp = b[perm]

    _, info0, it0, t0 = _cg(Ap, bp, rtol, maxiter)

    # ---- IC(0) ---------------------------------------------------------
    with Timer() as tfc:
        ic = ic0_factor(Ap)
    prec_ic = ic.as_operator()
    _, info_ic, it_ic, t_ic = _cg(Ap, bp, rtol, maxiter, prec_ic)
    # isolate the cost of applying the preconditioner (two triangular
    # solves), which is where cache locality shows up
    with Timer() as ta:
        for _ in range(n_apply):
            ic.solve(bp)
    t_apply = ta.t / n_apply

    out = {
        "ic0_nnz": ic.nnz, "ic0_fill_ratio": ic.nnz / Ap.nnz,
        "ic0_shift": ic.shift, "ic0_attempts": ic.attempts,
        "ic0_iters": it_ic, "ic0_converged": info_ic == 0,
        "t_ic0_factor": tfc.t, "t_ic0_solve": t_ic,
        "t_ic0_apply_once": t_apply,
        "t_total_ic0": t_reorder + tfc.t + t_ic,
    }

    # ---- threshold ILU, for comparison ---------------------------------
    if with_ilu:
        with Timer() as tf:
            ilu = _ilu(Ap, drop_tol)
        nnz_ilu = int(ilu.L.nnz + ilu.U.nnz)
        prec = spla.LinearOperator(Ap.shape, ilu.solve)
        _, info1, it1, t1 = _cg(Ap, bp, rtol, maxiter, prec)
        out.update({
            "ilu_nnz": nnz_ilu, "ilu_fill_ratio": nnz_ilu / Ap.nnz,
            "cg_iters": it1, "cg_converged": info1 == 0,
            "t_factor": tf.t, "t_solve": t1,
            "t_total_no_permute": t_reorder + tf.t + t1,
        })

    cache = simulate_cache_misses(Ap)
    g = gap_stats(A, perm)
    out.update({
        "bw_max": g["max"], "bw_mean": g["mean"],
        "bw_median": g["median"], "bw_p99": g["p99"],
        "cg_iters_plain": it0, "cg_converged_plain": info0 == 0,
        "cache_misses": cache["misses"], "cache_miss_rate": cache["miss_rate"],
        "t_reorder": t_reorder, "t_permute": tp.t, "t_solve_plain": t0,
    })
    return out


def run_one_m(vol, m, steps, args):
    run = model.DiffusionRun(vol, n_target=args.n_target, seed=args.seed,
                             tau=args.tau,
                             refine_fraction=args.refine_fraction)
    perms = {s: (np.arange(run.mesh.n_vertices), 0.0) for s in SCHEMES}
    stale = True
    per_step = []
    for s in range(steps):
        M, K, K0, g, gradnorm = run.operators()
        A = (M + args.tau * K).tocsr()
        b = M @ run.u
        n = A.shape[0]

        if stale:
            perms["natural"] = (np.arange(n), 0.0)
            pr, tr = order_rcm(run.mesh)
            perms["rcm"] = (pr, tr)
            ph, th = order_phve(run.mesh, args.order)
            perms["phve"] = (ph, th)
            stale = False
        else:
            # numbering reused; newly inserted nodes keep insertion labels
            for s_ in SCHEMES:
                p_old, _ = perms[s_]
                if p_old.size < n:
                    perms[s_] = (np.concatenate(
                        [p_old, np.arange(p_old.size, n)]), 0.0)
                else:
                    perms[s_] = (p_old[:n], 0.0)

        rec = {"step": s, "N": int(n), "nnz": int(A.nnz),
               "n_tets": run.mesh.n_tets, "remeshed_this_step": False}
        for name in SCHEMES:
            perm, t_re = perms[name]
            rec[name] = measure(A, b, perm, t_re, args.drop_tol, args.rtol,
                                with_ilu=not args.no_ilu)
            perms[name] = (perm, 0.0)   # cost paid once
        per_step.append(rec)

        run.u, _ = fem.step(M, K, args.tau, run.u)
        if (s + 1) % m == 0 and s + 1 < steps:
            n_new = run.remesh(gradnorm)
            if n_new:
                rec["remeshed_this_step"] = True
                stale = True
    return per_step


def totals(per_step):
    out = {}
    for name in SCHEMES:
        keys = ("t_reorder", "t_factor", "t_solve", "t_total_no_permute",
                "t_ic0_factor", "t_ic0_solve", "t_total_ic0",
                "t_ic0_apply_once")
        out[name] = {k: float(sum(r[name].get(k, 0.0) for r in per_step))
                     for k in keys}
        out[name]["ic0_iters_total"] = int(sum(r[name]["ic0_iters"]
                                               for r in per_step))
        out[name]["ic0_fill_ratio_mean"] = float(
            np.mean([r[name]["ic0_fill_ratio"] for r in per_step]))
        out[name]["ic0_max_shift"] = float(
            max(r[name]["ic0_shift"] for r in per_step))
        out[name]["cg_iters_total"] = int(sum(r[name].get("cg_iters", 0)
                                              for r in per_step))
        out[name]["cg_iters_plain_total"] = int(sum(r[name]["cg_iters_plain"]
                                                    for r in per_step))
        out[name]["n_steps_ilu_diverged"] = int(sum(
            0 if r[name].get("cg_converged", True) else 1 for r in per_step))
        out[name]["cache_misses_total"] = int(sum(r[name]["cache_misses"]
                                                  for r in per_step))
        out[name]["bw_max_mean_over_steps"] = float(
            np.mean([r[name]["bw_max"] for r in per_step]))
        out[name]["bw_mean_mean_over_steps"] = float(
            np.mean([r[name]["bw_mean"] for r in per_step]))
        out[name]["ilu_fill_ratio_mean"] = float(
            np.mean([r[name].get("ilu_fill_ratio", np.nan)
                     for r in per_step]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--remesh-every", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--refine-fraction", type=float, default=0.03)
    ap.add_argument("--no-ilu", action="store_true",
                    help="skip the threshold-ILU comparison (much faster); "
                         "IC(0) is the headline preconditioner anyway")
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--drop-tol", type=float, default=1e-4)
    ap.add_argument("--rtol", type=float, default=1e-10)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    results = {}
    for m in args.remesh_every:
        t0 = time.perf_counter()
        per_step = run_one_m(vol, m, args.steps, args)
        tot = totals(per_step)
        results[str(m)] = {"per_step": per_step, "totals": tot,
                           "wall_seconds": time.perf_counter() - t0,
                           "final_N": per_step[-1]["N"]}
        print("\n--- remesh every m=%d, %d steps, final N=%d ---"
              % (m, args.steps, per_step[-1]["N"]))
        for name in SCHEMES:
            t = tot[name]
            print("  %-8s IC0: renum %6.3f + fact %6.3f + solve %6.3f "
                  "= TOTAL %6.3f s | its %4d | apply %6.2f ms | fill x%.2f"
                  % (name, t["t_reorder"], t["t_ic0_factor"],
                     t["t_ic0_solve"], t["t_total_ic0"],
                     t["ic0_iters_total"], 1e3 * t["t_ic0_apply_once"],
                     t["ic0_fill_ratio_mean"]))
            if args.no_ilu:
                print("  %-8s      plainCG %5d | BWmax %7.0f BWmean %7.1f "
                      "| miss %.4g"
                      % ("", t["cg_iters_plain_total"],
                         t["bw_max_mean_over_steps"],
                         t["bw_mean_mean_over_steps"], t["cache_misses_total"]))
                continue
            print("  %-8s ILU: total %7.3f s | its %5d (%d diverged) "
                  "| fill x%.2f || plainCG %5d | BWmax %7.0f BWmean %7.1f "
                  "| miss %.4g"
                  % ("", t["t_total_no_permute"], t["cg_iters_total"],
                     t["n_steps_ilu_diverged"], t["ilu_fill_ratio_mean"],
                     t["cg_iters_plain_total"], t["bw_max_mean_over_steps"],
                     t["bw_mean_mean_over_steps"], t["cache_misses_total"]))

    out = {"experiment": "exp04_comparative", "params": vars(args),
           "hardware": hardware_record(), "by_remesh_period": results}
    save_json(out, "exp04_comparative.json")


if __name__ == "__main__":
    main()
