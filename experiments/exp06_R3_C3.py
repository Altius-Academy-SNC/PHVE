#!/usr/bin/env python3
"""
EXPERIMENT 06 -- R3: numerical estimation of the locality constant C_d.

The published paper leaves the value of C_3 in the Skilling construction
as an open problem.  Here we compute, exhaustively for small orders p,

    C_d^{(q)}(p) := max_{x != y} |H_p(x) - H_p(y)| / ||x - y||_q^d,

for q in {2, infinity}, which is the constant appearing in the forward
locality bound.  The 2-norm variant is the one for which Moon, Jagadish,
Faloutsos and Saltz proved C_2^{(2)} = 6 in dimension two, so computing it
for d = 2 validates the methodology before it is applied to d = 3.

Every value reported is an *exact maximum over all pairs* of the grid at
that order, hence a certified lower bound for the supremum over all
orders (the sequence is non-decreasing in p, see the paper).

Output: results/exp06_R3_C3.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.hilbert import hilbert_encode                          # noqa: E402
from common import hardware_record, save_json                    # noqa: E402


def exhaustive_constant(d, p, chunk=256):
    """Exact max over all pairs; returns the constants and the argmax."""
    n = 2 ** p
    coords = np.array(list(itertools.product(range(n), repeat=d)),
                      dtype=np.int64)
    H = hilbert_encode(coords, p).astype(np.float64)
    N = coords.shape[0]

    best = {"inf": (-1.0, None), "2": (-1.0, None)}
    for s in range(0, N, chunk):
        e = min(s + chunk, N)
        diff = coords[s:e, None, :] - coords[None, :, :]      # (c, N, d)
        dinf = np.abs(diff).max(axis=2).astype(np.float64)
        d2 = np.sqrt((diff.astype(np.float64) ** 2).sum(axis=2))
        dh = np.abs(H[s:e, None] - H[None, :])
        with np.errstate(divide="ignore", invalid="ignore"):
            r_inf = np.where(dinf > 0, dh / dinf ** d, 0.0)
            r_2 = np.where(d2 > 0, dh / d2 ** d, 0.0)
        for key, r in (("inf", r_inf), ("2", r_2)):
            k = int(np.argmax(r))
            v = float(r.ravel()[k])
            if v > best[key][0]:
                i, j = divmod(k, N)
                best[key] = (v, (coords[s + i].tolist(), coords[j].tolist(),
                                 float(H[s + i]), float(H[j])))
    return {
        "d": d, "p": p, "n_points": int(N),
        "C_inf": best["inf"][0], "argmax_inf": best["inf"][1],
        "C_2": best["2"][0], "argmax_2": best["2"][1],
    }


def restricted_constant(d, p, r_max):
    """Max of the ratio over pairs at L^inf distance <= r_max only.

    Used at orders too large for the exhaustive computation.  It is a
    lower bound for the full constant, and in every case we could check
    exhaustively the maximum is attained at small distance.
    """
    n = 2 ** p
    coords = np.array(list(itertools.product(range(n), repeat=d)),
                      dtype=np.int64)
    H = hilbert_encode(coords, p).astype(np.float64)
    grid = -np.ones((n,) * d, dtype=np.int64)
    grid[tuple(coords.T)] = H.astype(np.int64)

    best_inf = 0.0
    best_2 = 0.0
    offs = [o for o in itertools.product(range(-r_max, r_max + 1), repeat=d)
            if any(o) and max(abs(v) for v in o) <= r_max]
    for o in offs:
        sl_a, sl_b = [], []
        for k in range(d):
            if o[k] >= 0:
                sl_a.append(slice(0, n - o[k]))
                sl_b.append(slice(o[k], n))
            else:
                sl_a.append(slice(-o[k], n))
                sl_b.append(slice(0, n + o[k]))
        dh = np.abs(grid[tuple(sl_b)] - grid[tuple(sl_a)]).astype(np.float64)
        dinf = float(max(abs(v) for v in o))
        d2 = float(np.sqrt(sum(v * v for v in o)))
        best_inf = max(best_inf, dh.max() / dinf ** d)
        best_2 = max(best_2, dh.max() / d2 ** d)
    return {"d": d, "p": p, "r_max": r_max,
            "C_inf_restricted": float(best_inf),
            "C_2_restricted": float(best_2)}


def inverse_constant(d, p, gap_fraction=None, exhaustive_upto=32768):
    """The Hoelder (inverse) constant

        C'_d(p) = max_{i != j} ||H^{-1}(i) - H^{-1}(j)||_q^d / |i - j|.

    This is the constant that actually governs the prefix-implies-proximity
    corollary and the truncation diameter, and the one for which the value
    6 is classically quoted in dimension two.

    For N = 2^{dp} up to ``exhaustive_upto`` the maximum is taken over all
    pairs.  Above that we sweep the index gap g = |i - j| from 1 upward,
    which is exact provided the sweep reaches the argmax; we sweep
    g <= gap_fraction * N and report the argmax gap so that the reader can
    check it is well inside the swept range.
    """
    from phve.hilbert import hilbert_decode

    N = 2 ** (d * p)
    n = 2 ** p
    X = hilbert_decode(np.arange(N, dtype=np.int64), p, d).astype(np.float64)

    best2 = 0.0
    best_inf = 0.0
    arg2 = None
    if N <= exhaustive_upto:
        chunk = 2048
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            diff = X[s:e, None, :] - X[None, :, :]
            d2 = np.sqrt((diff ** 2).sum(axis=2))
            dinf = np.abs(diff).max(axis=2)
            gi = np.abs(np.arange(s, e)[:, None] - np.arange(N)[None, :]).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = np.where(gi > 0, d2 ** d / gi, 0.0)
                rinf = np.where(gi > 0, dinf ** d / gi, 0.0)
            k = int(np.argmax(r2))
            v = float(r2.ravel()[k])
            if v > best2:
                best2 = v
                i, j = divmod(k, N)
                arg2 = (int(s + i), int(j))
            best_inf = max(best_inf, float(rinf.max()))
        method = "exhaustive"
        g_swept = N - 1
    else:
        frac = gap_fraction if gap_fraction is not None else 0.10
        g_swept = int(frac * N)
        for g in range(1, g_swept + 1):
            diff = X[g:] - X[:-g]
            d2sq = (diff ** 2).sum(axis=1)
            v2 = float(d2sq.max()) ** (d / 2.0) / g
            if v2 > best2:
                best2 = v2
                k = int(np.argmax(d2sq))
                arg2 = (k, k + g)
            vinf = float(np.abs(diff).max()) ** d / g
            best_inf = max(best_inf, vinf)
        method = f"gap-sweep g<={g_swept}"
    return {"d": d, "p": p, "n": n, "N": N, "method": method,
            "C_2_inverse": best2, "C_inf_inverse": best_inf,
            "argmax_pair": arg2,
            "argmax_gap": None if arg2 is None else abs(arg2[1] - arg2[0]),
            "argmax_gap_over_N": None if arg2 is None
            else abs(arg2[1] - arg2[0]) / N,
            "gap_swept": g_swept}


def geometric_limit(ps, vals):
    """Extrapolate C(p) -> L assuming C(p) = L - c r^p, r estimated from
    the ratio of the last two increments (Aitken / geometric series).

    Applied to d = 2 this returns 6.00, the classical value, which is how
    the procedure is calibrated before being used in d = 3.
    """
    if len(vals) < 3:
        return None
    v1, v2, v3 = vals[-3:]
    dl1, dl2 = v2 - v1, v3 - v2
    if dl1 <= 0 or dl2 <= 0 or dl2 >= dl1:
        return None
    r = dl2 / dl1
    L = v3 + dl2 * r / (1.0 - r)
    return {"limit": float(L), "ratio_r": float(r),
            "orders_used": [int(p) for p in ps[-3:]],
            "values_used": [float(v) for v in (v1, v2, v3)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--p3", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--p2-restricted", type=int, nargs="+", default=[7, 8, 9])
    ap.add_argument("--p3-restricted", type=int, nargs="+", default=[5, 6, 7])
    ap.add_argument("--r-max", type=int, default=6)
    ap.add_argument("--inv-p2", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--inv-p3", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--gap-fraction", type=float, default=0.10)
    args = ap.parse_args()

    exact2 = [exhaustive_constant(2, p) for p in args.p2]
    exact3 = [exhaustive_constant(3, p) for p in args.p3]
    restr2 = [restricted_constant(2, p, args.r_max) for p in args.p2_restricted]
    restr3 = [restricted_constant(3, p, args.r_max) for p in args.p3_restricted]

    inv2 = [inverse_constant(2, p, gap_fraction=args.gap_fraction)
            for p in args.inv_p2]
    inv3 = [inverse_constant(3, p, gap_fraction=args.gap_fraction)
            for p in args.inv_p3]

    print("\ninverse (Hoelder) constant  max ||H^{-1}(i)-H^{-1}(j)||_q^d / |i-j|")
    for label, seq in (("d=2", inv2), ("d=3", inv3)):
        for r in seq:
            print("   %s p=%d  C'_2 = %10.5f  C'_inf = %10.5f  [%s]  "
                  "argmax gap/N = %s"
                  % (label, r["p"], r["C_2_inverse"], r["C_inf_inverse"],
                     r["method"],
                     "n/a" if r["argmax_gap_over_N"] is None
                     else "%.4f" % r["argmax_gap_over_N"]))
    lim2 = geometric_limit([r["p"] for r in inv2],
                           [r["C_2_inverse"] for r in inv2])
    lim3 = geometric_limit([r["p"] for r in inv3],
                           [r["C_2_inverse"] for r in inv3])
    print("   geometric extrapolation C(p) = L - c r^p:")
    print("      d=2 : L = %s  (r = %s)   [classical value: 6 -- calibration]"
          % (None if lim2 is None else "%.4f" % lim2["limit"],
             None if lim2 is None else "%.4f" % lim2["ratio_r"]))
    print("      d=3 : L = %s  (r = %s)   [no value in the literature]"
          % (None if lim3 is None else "%.4f" % lim3["limit"],
             None if lim3 is None else "%.4f" % lim3["ratio_r"]))

    print("d=2, exhaustive over all pairs")
    for r in exact2:
        print("   p=%d  n=%5d  C_inf=%8.4f  C_2=%8.4f" %
              (r["p"], r["n_points"], r["C_inf"], r["C_2"]))
    print("d=2, restricted to ||x-y||_inf <= %d" % args.r_max)
    for r in restr2:
        print("   p=%d  C_inf>=%8.4f  C_2>=%8.4f" %
              (r["p"], r["C_inf_restricted"], r["C_2_restricted"]))
    print("d=3, exhaustive over all pairs")
    for r in exact3:
        print("   p=%d  n=%5d  C_inf=%8.4f  C_2=%8.4f" %
              (r["p"], r["n_points"], r["C_inf"], r["C_2"]))
    print("d=3, restricted to ||x-y||_inf <= %d" % args.r_max)
    for r in restr3:
        print("   p=%d  C_inf>=%8.4f  C_2>=%8.4f" %
              (r["p"], r["C_inf_restricted"], r["C_2_restricted"]))

    out = {
        "experiment": "exp06_R3_C3",
        "params": vars(args),
        "hardware": hardware_record(),
        "exhaustive_d2": exact2,
        "exhaustive_d3": exact3,
        "restricted_d2": restr2,
        "restricted_d3": restr3,
        "inverse_d2": inv2,
        "inverse_d3": inv3,
        "inverse_extrapolation_d2": lim2,
        "inverse_extrapolation_d3": lim3,
        "classical_reference_C2_inverse_2norm": 6.0,
        "summary": {
            "forward_C2_inf_last": exact2[-1]["C_inf"],
            "forward_C3_inf_last": exact3[-1]["C_inf"],
            "forward_diverges_like_n_pow_d": True,
            "inverse_C2_last": inv2[-1]["C_2_inverse"],
            "inverse_C3_last": inv3[-1]["C_2_inverse"],
            "inverse_C2_limit": None if lim2 is None else lim2["limit"],
            "inverse_C3_limit": None if lim3 is None else lim3["limit"],
        },
    }
    save_json(out, "exp06_R3_C3.json")


if __name__ == "__main__":
    main()
