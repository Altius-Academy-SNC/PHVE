#!/usr/bin/env python3
"""
EXPERIMENT 17 -- Exhaustive verification of the closed forms for the
adjacent-pair maxima (extends U6).

The seam theorem proves the lower bound (1 - 2^{1-d}) n^d with an explicit
witness.  What is *conjectured* is the exact value

    A_2(p) = (5 * 4^p - 2) / 6,        A_3(p) = (13 * 8^p - 6) / 14,

i.e. that no adjacent pair beats the seam pair.  U6 records that these were
checked exhaustively only to p = 6 in d = 2 and p = 4 in d = 3, which is
thin support for what is now the paper's strongest new statement.

The check is much cheaper than it looks and does not need the O(N^2) sweep
the inverse constant requires.  A(p) is a maximum over *adjacent* pairs
only, of which there are d(n-1)n^{d-1} = O(d n^d).  Enumerating them by
shifting one axis at a time, and encoding in chunks, the cost is linear in
the number of grid points.

Both variants are checked, because exp13 showed they differ in d = 3 and
the closed form is a claim about the Skilling curve specifically.

Output: results/exp17_seam_closed_forms.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.compact_hilbert import hilbert_index_hrc                # noqa: E402
from phve.hilbert import hilbert_encode                            # noqa: E402
from common import hardware_record, save_json                      # noqa: E402


def closed_form(d, p):
    if d == 2:
        return (5 * 4 ** p - 2) // 6
    if d == 3:
        return (13 * 8 ** p - 6) // 14
    return None


def seam_lower_bound(d, p):
    """(1 - 2^{1-d}) n^d, the proved bound."""
    n = 1 << p
    return (1.0 - 2.0 ** (1 - d)) * n ** d


def grid_chunks(d, p, chunk):
    """Yield chunks of all points of {0,..,2^p-1}^d in odometer order."""
    n = 1 << p
    total = n ** d
    for s in range(0, total, chunk):
        e = min(s + chunk, total)
        lin = np.arange(s, e, dtype=np.int64)
        G = np.empty((e - s, d), dtype=np.int64)
        rem = lin
        for j in range(d - 1, -1, -1):
            G[:, j] = rem % n
            rem = rem // n
        yield G


def adjacent_max_skilling(d, p, chunk=1 << 21):
    """max |H(x) - H(x + e_j)| over all axes j and all valid x."""
    n = 1 << p
    best = 0
    witness = None
    for G in grid_chunks(d, p, chunk):
        h = hilbert_encode(G, p)
        for j in range(d):
            keep = G[:, j] < n - 1
            if not keep.any():
                continue
            G2 = G[keep].copy()
            G2[:, j] += 1
            h2 = hilbert_encode(G2, p)
            gaps = np.abs(h[keep] - h2)
            k = int(np.argmax(gaps))
            if int(gaps[k]) > best:
                best = int(gaps[k])
                witness = (G[keep][k].tolist(), G2[k].tolist())
    return best, witness


def adjacent_max_hrc(d, p):
    """Same, for the H&RC variant.  Scalar, so only small p."""
    n = 1 << p
    from itertools import product
    pts = np.array(list(product(*([range(n)] * d))), dtype=np.int64)
    idx = {tuple(row): hilbert_index_hrc(row, d, p) for row in pts}
    best = 0
    witness = None
    for row in pts:
        t = tuple(row)
        for j in range(d):
            if row[j] < n - 1:
                u = list(t)
                u[j] += 1
                g = abs(idx[t] - idx[tuple(u)])
                if g > best:
                    best = g
                    witness = (list(t), u)
    return best, witness


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ap.add_argument("--p3", type=int, nargs="+",
                    default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--hrc-p2", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--hrc-p3", type=int, nargs="+", default=[1, 2, 3, 4])
    args = ap.parse_args()

    out = {"experiment": "exp17_seam_closed_forms", "params": vars(args),
           "hardware": hardware_record(), "skilling": {}, "hrc": {}}

    for d, ps in ((2, args.p2), (3, args.p3)):
        rows = []
        print("=== Skilling, d = %d ===" % d)
        print("   p | grid points |      A(p) measured |    closed form | ok |"
              " lower bound | A/n^d")
        for p in ps:
            if d * p > 60:
                continue
            npts = (1 << p) ** d
            a, w = adjacent_max_skilling(d, p)
            cf = closed_form(d, p)
            lb = seam_lower_bound(d, p)
            ok = (cf is not None and a == cf)
            rows.append({"p": p, "n_points": int(npts), "A": int(a),
                         "closed_form": int(cf) if cf else None,
                         "matches_closed_form": bool(ok),
                         "seam_lower_bound": lb,
                         "above_lower_bound": bool(a >= lb),
                         "A_over_n_d": a / npts, "witness": w})
            print("  %2d | %11d | %18d | %14s | %-3s | %11.1f | %.4f"
                  % (p, npts, a, cf, "yes" if ok else "NO", lb, a / npts))
            sys.stdout.flush()
        out["skilling"]["d%d" % d] = rows

    for d, ps in ((2, args.hrc_p2), (3, args.hrc_p3)):
        rows = []
        print("\n=== Hamilton & Rau-Chaplin, d = %d ===" % d)
        print("   p | grid points |      A(p) measured | Skilling closed form"
              " | same? |")
        for p in ps:
            npts = (1 << p) ** d
            a, w = adjacent_max_hrc(d, p)
            cf = closed_form(d, p)
            rows.append({"p": p, "n_points": int(npts), "A": int(a),
                         "skilling_closed_form": int(cf) if cf else None,
                         "same_as_skilling": bool(cf is not None and a == cf),
                         "witness": w})
            print("  %2d | %11d | %18d | %20s | %-5s |"
                  % (p, npts, a, cf, "yes" if a == cf else "NO"))
            sys.stdout.flush()
        out["hrc"]["d%d" % d] = rows

    print("\n=== Verdict ===")
    for d in (2, 3):
        rows = out["skilling"]["d%d" % d]
        allok = all(r["matches_closed_form"] for r in rows)
        top = max(r["p"] for r in rows)
        print("   Skilling d=%d: closed form holds exhaustively up to p = %d"
              "  -> %s" % (d, top, "confirmed" if allok else "REFUTED"))
        if not allok:
            bad = [r["p"] for r in rows if not r["matches_closed_form"]]
            print("      fails at p = %s" % bad)
        print("   Skilling d=%d: A(p)/n^d -> %.6f  (predicted %.6f)"
              % (d, rows[-1]["A_over_n_d"],
                 (5 / 6) if d == 2 else (13 / 14)))
    out["verdict"] = {
        "skilling_closed_form_confirmed": {
            "d%d" % d: all(r["matches_closed_form"]
                           for r in out["skilling"]["d%d" % d])
            for d in (2, 3)},
        "max_p_verified": {"d%d" % d: max(r["p"] for r in
                                          out["skilling"]["d%d" % d])
                           for d in (2, 3)},
    }

    save_json(out, "exp17_seam_closed_forms.json")


if __name__ == "__main__":
    main()
