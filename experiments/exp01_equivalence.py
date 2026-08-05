#!/usr/bin/env python3
"""
EXPERIMENT 01 -- Numerical verification of the Equivalence Lemma.

Claim under test (Lemma "renumbering is analytically inert"):
let P be the permutation matrix induced by the PHVE order on the degrees
of freedom.  Then

    (P M P^T + tau P K P^T) v^{n+1} = P M P^T v^n,    v = P u

has the same spectrum, the same condition number in every unitarily
invariant norm, and the solution v^{n+1} = P u^{n+1}.

What is actually measurable: the *exact* arithmetic statement is exact,
but floating-point summation is not associative, so the computed solutions
differ.  This script measures that difference and checks that it stays at
the level of round-off accumulation.

Usage:  python exp01_equivalence.py [--n-target N] [--seed S]
Output: results/exp01_equivalence.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                    # noqa: E402
from phve.metrics import (apply_permutation, bandwidth,        # noqa: E402
                          order_natural, order_phve, order_rcm)
from common import RESULTS, hardware_record, save_json         # noqa: E402


def spectral_check(A, B, k_dense=3000):
    """Compare the spectra of A and B = P A P^T."""
    n = A.shape[0]
    if n <= k_dense:
        ea = np.linalg.eigvalsh(A.toarray())
        eb = np.linalg.eigvalsh(B.toarray())
        scale = max(abs(ea).max(), 1.0)
        return {
            "method": "dense",
            "n": int(n),
            "max_abs_eig_diff": float(np.abs(ea - eb).max()),
            "max_rel_eig_diff": float(np.abs(ea - eb).max() / scale),
            "lambda_min_A": float(ea[0]), "lambda_max_A": float(ea[-1]),
            "lambda_min_B": float(eb[0]), "lambda_max_B": float(eb[-1]),
            "cond_A": float(ea[-1] / ea[0]) if ea[0] > 0 else float("inf"),
            "cond_B": float(eb[-1] / eb[0]) if eb[0] > 0 else float("inf"),
        }
    ea_hi = spla.eigsh(A, k=4, which="LA", return_eigenvectors=False)
    eb_hi = spla.eigsh(B, k=4, which="LA", return_eigenvectors=False)
    ea_lo = spla.eigsh(A, k=4, which="SA", return_eigenvectors=False)
    eb_lo = spla.eigsh(B, k=4, which="SA", return_eigenvectors=False)
    return {
        "method": "lanczos-extremes",
        "n": int(n),
        "max_abs_eig_diff": float(max(np.abs(np.sort(ea_hi) - np.sort(eb_hi)).max(),
                                      np.abs(np.sort(ea_lo) - np.sort(eb_lo)).max())),
        "lambda_min_A": float(ea_lo.min()), "lambda_max_A": float(ea_hi.max()),
        "lambda_min_B": float(eb_lo.min()), "lambda_max_B": float(eb_hi.max()),
        "cond_A": float(ea_hi.max() / ea_lo.min()),
        "cond_B": float(eb_hi.max() / eb_lo.min()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--order", type=int, default=8)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--tau", type=float, default=1.0)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    run = model.DiffusionRun(vol, n_target=args.n_target, seed=args.seed,
                             tau=args.tau)
    M, K, K0, g, gradnorm = run.operators()
    n = M.shape[0]

    perm_nat, _ = order_natural(run.mesh)
    perm_h, t_h = order_phve(run.mesh, args.order)
    perm_r, t_r = order_rcm(run.mesh)

    A = (M + args.tau * K).tocsr()
    b = M @ run.u

    out = {
        "experiment": "exp01_equivalence",
        "params": vars(args),
        "hardware": hardware_record(),
        "mesh": {"n_vertices": n, "n_tets": run.mesh.n_tets,
                 "nnz_A": int(A.nnz)},
        "g_min": float(g.min()), "g_max": float(g.max()),
    }

    # ---- (i) spectrum and condition number -----------------------------
    A_h = apply_permutation(A, perm_h)
    A_r = apply_permutation(A, perm_r)
    out["spectrum_phve"] = spectral_check(A, A_h)
    out["spectrum_rcm"] = spectral_check(A, A_r)

    # ---- (ii) bandwidth actually changes -------------------------------
    out["bandwidth"] = {
        "natural": bandwidth(A),
        "rcm": bandwidth(A_r),
        "phve": bandwidth(A_h),
    }
    out["reorder_seconds"] = {"rcm": t_r, "phve": t_h}

    # ---- (iii) one step, direct solve ----------------------------------
    inv_h = np.empty_like(perm_h); inv_h[perm_h] = np.arange(n)
    x_nat = spla.spsolve(A.tocsc(), b)
    x_h = spla.spsolve(A_h.tocsc(), b[perm_h])
    diff = x_h - x_nat[perm_h]
    out["one_step_direct"] = {
        "norm_diff_2": float(np.linalg.norm(diff)),
        "norm_diff_inf": float(np.abs(diff).max()),
        "rel_diff_2": float(np.linalg.norm(diff) / np.linalg.norm(x_nat)),
        "rel_diff_inf": float(np.abs(diff).max() / np.abs(x_nat).max()),
        "eps_machine": float(np.finfo(float).eps),
        "rel_diff_in_eps": float(np.linalg.norm(diff) /
                                 np.linalg.norm(x_nat) / np.finfo(float).eps),
    }

    # ---- (iv) full trajectory, iterative solve -------------------------
    u_nat = run.u.copy()
    u_h = run.u[perm_h].copy()
    Mh = apply_permutation(M, perm_h)
    Kh = apply_permutation(K, perm_h)
    traj = []
    for s in range(args.steps):
        u_nat, it1 = fem.step(M, K, args.tau, u_nat, rtol=1e-12)
        u_h, it2 = fem.step(Mh, Kh, args.tau, u_h, rtol=1e-12)
        d = u_h - u_nat[perm_h]
        traj.append({
            "step": s + 1,
            "cg_iters_natural": it1, "cg_iters_phve": it2,
            "rel_diff_2": float(np.linalg.norm(d) / np.linalg.norm(u_nat)),
            "rel_diff_inf": float(np.abs(d).max() / np.abs(u_nat).max()),
        })
    out["trajectory"] = traj
    out["trajectory_summary"] = {
        "final_rel_diff_2": traj[-1]["rel_diff_2"],
        "final_rel_diff_2_in_eps": traj[-1]["rel_diff_2"] / np.finfo(float).eps,
        "cg_iters_identical": all(t["cg_iters_natural"] == t["cg_iters_phve"]
                                  for t in traj),
        "max_cg_iter_gap": max(abs(t["cg_iters_natural"] - t["cg_iters_phve"])
                               for t in traj),
    }

    save_json(out, "exp01_equivalence.json")
    print(json.dumps({k: out[k] for k in
                      ("mesh", "bandwidth", "spectrum_phve",
                       "one_step_direct", "trajectory_summary")}, indent=2))


if __name__ == "__main__":
    main()
