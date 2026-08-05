#!/usr/bin/env python3
"""
EXPERIMENT 09 -- The two quantities of the comparative study that are not
wall-clock times, swept over N.

The timings of exp04 are not reliable at the granularity we need: repeated
single runs disagree with each other by more than the effect being
measured.  Two of the quantities involved are, however, *deterministic*:

  (a) the simulated cache misses of one CSR sparse matrix--vector product,
      which is an exact function of the ordering and the cache geometry;
  (b) the IC(0)-preconditioned conjugate gradient iteration count, which
      depends only on the matrix and the ordering, not on the machine.

Both are swept over N, and (a) over three cache sizes, because the
expectation is that the locality advantage of a space-filling curve appears
only once the working set exceeds the cache.

Output: results/exp09_cache_and_ic0.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import fem, model                                      # noqa: E402
from phve.ic0 import ic0_factor                                  # noqa: E402
from phve.metrics import (apply_permutation, order_phve,          # noqa: E402
                          order_rcm, simulate_cache_misses)
from common import FIGURES, hardware_record, save_json           # noqa: E402

SCHEMES = ("natural", "rcm", "phve")

# (label, sets, ways, line) -> total size = sets*ways*line
CACHES = [
    ("L1 32 KiB 8-way", 64, 8, 64),
    ("L2 1 MiB 16-way", 1024, 16, 64),
    ("L3 16 MiB 16-way", 16384, 16, 64),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[2000, 4000, 8000, 16000, 32000, 64000])
    ap.add_argument("--order", type=int, default=10)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--rtol", type=float, default=1e-10)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    records = []
    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed,
                                     tau=args.tau)
            M, K, K0, g, gradnorm = run.operators()
            run.u, _ = fem.step(M, K, args.tau, run.u)
            run.remesh(gradnorm)
            M, K, K0, g, gradnorm = run.operators()
            A = (M + args.tau * K).tocsr()
            b = M @ run.u
            n = A.shape[0]

            perms = {"natural": np.arange(n),
                     "rcm": order_rcm(run.mesh)[0],
                     "phve": order_phve(run.mesh, args.order)[0]}
            rec = {"seed": seed, "N": int(n), "nnz": int(A.nnz),
                   "working_set_kib": 8.0 * n / 1024.0}
            for name, perm in perms.items():
                Ap = apply_permutation(A, perm).tocsr()
                bp = b[perm]
                ic = ic0_factor(Ap)
                it = {"n": 0}
                _, info = spla.cg(Ap, bp, rtol=args.rtol, atol=0.0,
                                  maxiter=5000, M=ic.as_operator(),
                                  callback=lambda z: it.__setitem__("n", it["n"] + 1))
                entry = {"ic0_iters": it["n"], "ic0_converged": info == 0,
                         "ic0_nnz": ic.nnz, "ic0_shift": ic.shift}
                for label, sets, ways, line in CACHES:
                    c = simulate_cache_misses(Ap, line=line, sets=sets,
                                              ways=ways)
                    entry[label] = {"misses": c["misses"],
                                    "miss_rate": c["miss_rate"]}
                rec[name] = entry
            records.append(rec)
            print("seed=%d N=%6d ws=%7.1f KiB | IC0 its: nat %4d rcm %4d phve %4d"
                  " | L1 misses: rcm %8d phve %8d  (phve/rcm %.3f)"
                  % (seed, n, rec["working_set_kib"],
                     rec["natural"]["ic0_iters"], rec["rcm"]["ic0_iters"],
                     rec["phve"]["ic0_iters"],
                     rec["rcm"][CACHES[0][0]]["misses"],
                     rec["phve"][CACHES[0][0]]["misses"],
                     rec["phve"][CACHES[0][0]]["misses"]
                     / max(rec["rcm"][CACHES[0][0]]["misses"], 1)))

    # ---- summaries -------------------------------------------------------
    summary = {}
    for label, *_ in CACHES:
        rows = []
        for r in records:
            rows.append({"N": r["N"], "seed": r["seed"],
                         "phve_over_rcm": r["phve"][label]["misses"]
                         / max(r["rcm"][label]["misses"], 1),
                         "phve": r["phve"][label]["misses"],
                         "rcm": r["rcm"][label]["misses"],
                         "natural": r["natural"][label]["misses"]})
        # crossover: smallest N at which phve/rcm < 1 for every seed above it
        Ns = sorted({row["N"] for row in rows})
        crossover = None
        for N0 in Ns:
            above = [row for row in rows if row["N"] >= N0]
            if above and all(row["phve_over_rcm"] < 1.0 for row in above):
                crossover = N0
                break
        summary[label] = {"rows": rows, "crossover_N": crossover}
        print("\n%s : PHVE/RCM miss ratio" % label)
        for row in sorted(rows, key=lambda x: (x["N"], x["seed"])):
            print("   N=%6d seed=%d  ratio=%.3f" % (row["N"], row["seed"],
                                                    row["phve_over_rcm"]))
        print("   crossover: PHVE better for N >= %s" % crossover)

    it_ratio = [(r["N"], r["phve"]["ic0_iters"] / max(r["rcm"]["ic0_iters"], 1))
                for r in records]
    print("\nIC(0) iteration ratio PHVE/RCM: " +
          ", ".join("%.2f" % v for _, v in it_ratio))
    print("   mean %.3f, min %.3f, max %.3f"
          % (float(np.mean([v for _, v in it_ratio])),
             float(np.min([v for _, v in it_ratio])),
             float(np.max([v for _, v in it_ratio]))))

    out = {"experiment": "exp09_cache_and_ic0", "params": vars(args),
           "hardware": hardware_record(), "caches": [c[0] for c in CACHES],
           "records": records, "summary": summary,
           "ic0_iteration_ratio_phve_over_rcm":
               [{"N": N, "ratio": v} for N, v in it_ratio]}
    save_json(out, "exp09_cache_and_ic0.json")


if __name__ == "__main__":
    main()
