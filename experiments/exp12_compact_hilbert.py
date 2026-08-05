#!/usr/bin/env python3
"""
EXPERIMENT 12 -- Does our anisotropic order really coincide with the
compact Hilbert index of Hamilton & Rau-Chaplin?  (closes U5)

The docstring of `phve/hilbert.py:phve_order_aniso` states that the
restricted-cube construction is "order-equivalent to the compact Hilbert
index of Hamilton & Rau-Chaplin (2008)".  Nothing verified it, and
`UNVERIFIED.md` U5 flags exactly this.  A claim of that kind sitting in the
source as if it were established is precisely what has to be removed.

Three separate questions are tested, deliberately kept apart because
conflating them is what makes this easy to get wrong:

  (Q1) *Which variant do we implement?*  Is our Skilling kernel's order on
       the cube the same total order as H&RC's own standard Hilbert index?
       There are 10 694 807 structurally distinct 3D Hilbert curves
       (Haverkort), so this is a real question, not a formality.

  (Q2) *Is compactness order-preserving?*  Within H&RC's own variant, does
       sorting by the compact index with per-axis bits m_j give the same
       total order as sorting by the standard index on the enclosing cube
       of order M = max_j m_j?  This is the rank-compression lemma, tested
       in the setting where it is stated.

  (Q3) *The claim as written.*  Does our `phve_order_aniso` induce the same
       total order as the compact Hilbert index?

Q3 can only hold if both Q1 and Q2 do, so the decomposition also localises
any failure.

Self-tests run first: the compact index with all m_j = M must reproduce
the standard index exactly, and must be a bijection onto
{0, ..., 2^{sum m_j} - 1}.  If either fails, the reference implementation
is wrong and nothing below means anything.

Output: results/exp12_compact_hilbert.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.compact_hilbert import (compact_hilbert_index,      # noqa: E402
                                  hilbert_index_hrc)
from phve.hilbert import hilbert_encode                        # noqa: E402
from common import hardware_record, save_json                  # noqa: E402


def full_grid(m):
    """All integer points of prod_j {0,...,2^{m_j}-1}, in lexicographic
    order (the enumeration order is irrelevant, only the induced sort is)."""
    axes = [np.arange(1 << mj, dtype=np.int64) for mj in m]
    return np.array(list(itertools.product(*axes)), dtype=np.int64)


def order_of(values):
    """Rank vector of a sequence, ties broken by position (stable)."""
    return np.argsort(np.asarray(values, dtype=object), kind="stable")


def same_total_order(a, b):
    """Do two index vectors induce the same total order on the same points?

    Both are injective here, so it suffices that the sorting permutations
    agree.
    """
    return bool(np.array_equal(order_of(a), order_of(b)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--cube-orders", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--aniso-cases", type=str, nargs="+",
                    default=["2,1", "3,1", "3,2", "4,2", "2,1,1", "3,2,1",
                             "4,2,2", "3,3,1", "4,3,2"],
                    help="comma-separated per-axis bit counts m_j")
    args = ap.parse_args()

    out = {"experiment": "exp12_compact_hilbert", "params": vars(args),
           "hardware": hardware_record()}

    # ------------------------------------------------------------------
    # Self-test 1: compact index with all m_j = M reproduces Algorithm 1
    # ------------------------------------------------------------------
    print("=== Self-test 1: compact(m_j = M) == standard H&RC index ===")
    st1 = []
    ok_all = True
    for n in args.dims:
        for M in args.cube_orders:
            if n * M > 22:
                continue
            G = full_grid([M] * n)
            a = [compact_hilbert_index(p, [M] * n, n) for p in G]
            b = [hilbert_index_hrc(p, n, M) for p in G]
            ok = a == b
            ok_all &= ok
            st1.append({"d": n, "M": M, "points": len(G), "identical": ok})
            print("   d=%d M=%d  %6d points  identical: %s" % (n, M, len(G), ok))
    out["selftest_compact_reduces_to_standard"] = {"rows": st1, "all_ok": ok_all}

    # ------------------------------------------------------------------
    # Self-test 2: bijectivity of the compact index
    # ------------------------------------------------------------------
    print("\n=== Self-test 2: compact index is a bijection onto [0, 2^sum m) ===")
    st2 = []
    ok_all2 = True
    for case in args.aniso_cases:
        m = [int(x) for x in case.split(",")]
        n = len(m)
        if sum(m) > 20:
            continue
        G = full_grid(m)
        h = [compact_hilbert_index(p, m, n) for p in G]
        expect = 1 << sum(m)
        ok = (len(set(h)) == len(h) == expect
              and min(h) == 0 and max(h) == expect - 1)
        ok_all2 &= ok
        st2.append({"m": m, "points": len(G), "range": expect,
                    "bijective_onto_range": ok})
        print("   m=%-10s %6d points  bijective onto [0,%d): %s"
              % (case, len(G), expect, ok))
    out["selftest_compact_bijective"] = {"rows": st2, "all_ok": ok_all2}

    if not (ok_all and ok_all2):
        print("\n!! reference implementation failed its own self-tests;"
              " the comparisons below are meaningless")
        save_json(out, "exp12_compact_hilbert.json")
        return 1

    # ------------------------------------------------------------------
    # Q1: is our Skilling variant the same curve as H&RC's?
    # ------------------------------------------------------------------
    print("\n=== Q1: Skilling cube order vs H&RC cube order ===")
    q1 = []
    for n in args.dims:
        for M in args.cube_orders:
            if n * M > 22:
                continue
            G = full_grid([M] * n)
            sk = hilbert_encode(G, M)
            hr = [hilbert_index_hrc(p, n, M) for p in G]
            identical_index = bool(np.array_equal(np.asarray(sk),
                                                  np.asarray(hr, dtype=np.int64)))
            same_order = same_total_order(sk, hr)
            q1.append({"d": n, "M": M, "identical_index": identical_index,
                       "same_total_order": same_order})
            print("   d=%d M=%d  same index: %-5s  same order: %-5s"
                  % (n, M, identical_index, same_order))
    out["Q1_skilling_vs_hrc_cube"] = q1

    # ------------------------------------------------------------------
    # Q2: within H&RC's variant, is the restricted-cube order the compact
    #     order?  (the rank-compression lemma, in its own setting)
    # ------------------------------------------------------------------
    print("\n=== Q2: H&RC restricted-cube order vs H&RC compact order ===")
    q2 = []
    for case in args.aniso_cases:
        m = [int(x) for x in case.split(",")]
        n = len(m)
        M = max(m)
        if sum(m) > 20 or n * M > 22:
            continue
        G = full_grid(m)
        comp = [compact_hilbert_index(p, m, n) for p in G]
        cube = [hilbert_index_hrc(p, n, M) for p in G]
        same = same_total_order(comp, cube)
        q2.append({"m": m, "M": M, "points": len(G), "same_total_order": same})
        print("   m=%-10s M=%d  %6d points  same order: %s"
              % (case, M, len(G), same))
    out["Q2_hrc_restricted_cube_vs_compact"] = q2

    # ------------------------------------------------------------------
    # Q3: the claim as written in phve/hilbert.py
    # ------------------------------------------------------------------
    print("\n=== Q3: our phve_order_aniso vs H&RC compact order ===")
    q3 = []
    for case in args.aniso_cases:
        m = [int(x) for x in case.split(",")]
        n = len(m)
        M = max(m)
        if sum(m) > 20 or n * M > 22:
            continue
        G = full_grid(m)
        # our construction: embed in the cube of order M, Skilling index
        ours = hilbert_encode(G, M)
        comp = [compact_hilbert_index(p, m, n) for p in G]
        same = same_total_order(ours, comp)
        q3.append({"m": m, "M": M, "points": len(G), "same_total_order": same})
        print("   m=%-10s M=%d  %6d points  same order: %s"
              % (case, M, len(G), same))
    out["Q3_ours_vs_compact"] = q3

    # ------------------------------------------------------------------
    print("\n=== Verdict ===")
    v = {
        "Q1_all_same_order": all(r["same_total_order"] for r in q1),
        "Q1_all_same_index": all(r["identical_index"] for r in q1),
        "Q2_all_same_order": all(r["same_total_order"] for r in q2),
        "Q3_all_same_order": all(r["same_total_order"] for r in q3),
    }
    for k, val in v.items():
        print("   %-22s %s" % (k, val))
    out["verdict"] = v

    save_json(out, "exp12_compact_hilbert.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
