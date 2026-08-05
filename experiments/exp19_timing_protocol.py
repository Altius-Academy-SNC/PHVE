#!/usr/bin/env python3
"""
EXPERIMENT 19 -- A timing protocol that can actually be quoted (blocking
item 1 of DIRECTION.md Sec. 6).

The problem
-----------
`exp04` reported wall-clock totals from *single* runs, and Sec. 2.6 of
`DIRECTION.md` records the consequence: the four remeshing frequencies
disagreed on the winner (RCM, PHVE, RCM, PHVE) and per-iteration times
varied by +-40 % between runs.  Nothing about total time could be claimed.

Two things were conflated there, and separating them is most of the fix:

  * quantities that are **deterministic** -- iteration counts, simulated
    cache misses, fill, index-gap functionals -- which need no repetition
    at all and are already reported elsewhere;
  * quantities that are **timings**, which need repetition, a reported
    spread, and a statement of whether the difference exceeds the noise.

This script does only the second, and only for the one timing the paper
actually needs: **the cost of computing the ordering**.  That is the term
that does not cancel under the equivalence lemma, and it is the one place
where "no adjacency graph" can pay for itself.

Protocol
--------
For each mesh size, the two orderings are timed `--repeats` times,
alternating between them so that any slow drift of the machine affects both
equally.  A discarded warm-up pass runs first.  We report the median, the
interquartile range, and the minimum (which is the least noisy estimator of
a lower bound on the true cost), and we declare a winner only when the
interquartile ranges do not overlap.

Nothing here is claimed to be machine-independent.  The point is to state a
timing with its spread instead of a single number, and to be explicit when
the spread swallows the effect.

Output: results/exp19_timing_protocol.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import model                                             # noqa: E402
from phve.metrics import order_phve, order_rcm                     # noqa: E402
from common import hardware_record, save_json                      # noqa: E402


def summarise(ts):
    a = np.sort(np.asarray(ts, dtype=float))
    return {"n": int(a.size), "median": float(np.median(a)),
            "q1": float(np.percentile(a, 25)),
            "q3": float(np.percentile(a, 75)),
            "min": float(a.min()), "max": float(a.max()),
            "iqr": float(np.percentile(a, 75) - np.percentile(a, 25)),
            "rel_iqr": float((np.percentile(a, 75) - np.percentile(a, 25))
                             / max(np.median(a), 1e-300))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+",
                    default=[8000, 16000, 32000, 64000, 128000])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--order", type=int, default=10)
    args = ap.parse_args()

    vol = model.load_mni152(2)
    records = []

    for seed in args.seeds:
        for n_target in args.sizes:
            run = model.DiffusionRun(vol, n_target=n_target, seed=seed)
            mesh = run.mesh
            n = mesh.n_vertices

            # warm-up, discarded
            order_rcm(mesh)
            order_phve(mesh, args.order)

            t_rcm, t_phve = [], []
            for _ in range(args.repeats):
                _, t = order_rcm(mesh)
                t_rcm.append(t)
                _, t = order_phve(mesh, args.order)
                t_phve.append(t)

            s_rcm = summarise(t_rcm)
            s_phve = summarise(t_phve)
            # a winner only if the interquartile ranges are disjoint
            separated = (s_rcm["q1"] > s_phve["q3"]) or (s_phve["q1"] > s_rcm["q3"])
            rec = {"seed": seed, "N": int(n), "repeats": args.repeats,
                   "rcm": s_rcm, "phve": s_phve,
                   "speedup_median": s_rcm["median"] / max(s_phve["median"], 1e-300),
                   "speedup_min": s_rcm["min"] / max(s_phve["min"], 1e-300),
                   "iqr_separated": bool(separated)}
            records.append(rec)
            print("seed=%d N=%6d | rcm %.4f s [IQR %.1f %%] | phve %.4f s"
                  " [IQR %.1f %%] | median speedup x%.2f | separated: %s"
                  % (seed, n, s_rcm["median"], 100 * s_rcm["rel_iqr"],
                     s_phve["median"], 100 * s_phve["rel_iqr"],
                     rec["speedup_median"], separated))
            sys.stdout.flush()

    print("\n=== Summary ===")
    sep = [r for r in records if r["iqr_separated"]]
    print("  configurations where the interquartile ranges are disjoint:"
          " %d of %d" % (len(sep), len(records)))
    if sep:
        sp = [r["speedup_median"] for r in sep]
        print("  median speedup of PHVE over RCM, over those: min %.2f max %.2f"
              % (min(sp), max(sp)))
    worst = max(records, key=lambda r: max(r["rcm"]["rel_iqr"],
                                           r["phve"]["rel_iqr"]))
    print("  worst relative IQR anywhere: %.1f %% (N = %d)"
          % (100 * max(worst["rcm"]["rel_iqr"], worst["phve"]["rel_iqr"]),
             worst["N"]))

    out = {"experiment": "exp19_timing_protocol", "params": vars(args),
           "hardware": hardware_record(), "records": records,
           "n_separated": len(sep), "n_total": len(records)}
    save_json(out, "exp19_timing_protocol.json")


if __name__ == "__main__":
    main()
