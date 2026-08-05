#!/usr/bin/env python3
"""
EXPERIMENT 14 -- Calibrating the WL_2 extrapolation against a second known
value (strengthens/replaces the R3 claim; see PRIOR_ART.md, corrected entry).

The situation
-------------
`exp06` computes the inverse locality constant

    WL_2(p) = max_{i != j} || H^{-1}(i) - H^{-1}(j) ||_2^d / |i - j|

for the Skilling variant and extrapolates WL_2(p) -> L geometrically.  In
d = 2 the procedure returns 5.999, recovering the classical value 6 of
Moon et al., which is the only calibration it had.  In d = 3 it returns
29.5, and `UNVERIFIED.md` U6 correctly flagged that an extrapolation is not
a proof.

The literature check then found published three-dimensional values:
Gotsman & Lindenbaum report WL_2 ~ 23 by simulation, and Haverkort
identifies the curve concerned as `A26.0010 1011.1011 0011` with
WL_2 = 22.9.  `exp12` and `exp13` established that Skilling's curve and
Hamilton & Rau-Chaplin's are *different* three-dimensional Hilbert curves
-- not related by any of the 96 cube symmetries (with or without traversal
reversal) for p >= 2 -- and that H&RC's has strictly better locality at
every order tested.

That converts a problem into a test.  If the same extrapolation procedure,
applied to the H&RC variant, returns ~22.9, then the procedure is
calibrated against *two independent published values* in *two different
dimensions*, and the Skilling figure it produces is credible.  If it
returns something else, the procedure is not trustworthy and the d = 3
number must be dropped.

This script runs exactly that test.  The extrapolation function is imported
from `exp06_R3_C3` rather than reimplemented, so there is no possibility of
tuning it to the desired answer.

Output: results/exp14_wl2_extrapolation.json
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve.compact_hilbert import hilbert_index_hrc               # noqa: E402
from phve.hilbert import hilbert_decode                           # noqa: E402
from exp06_R3_C3 import geometric_limit                           # noqa: E402
from common import hardware_record, save_json                     # noqa: E402

PUBLISHED = {
    ("skilling", 2): {"value": 6.0, "source": "Moon et al., IEEE TKDE 13 (2001)"},
    ("hrc", 2): {"value": 6.0, "source": "Moon et al., IEEE TKDE 13 (2001)"},
    ("hrc", 3): {"value": 22.9,
                 "source": "Haverkort, arXiv:1109.2323 (curve A26.0010 1011.1011 0011); "
                           "Gotsman & Lindenbaum 1996 report ~23 by simulation"},
}


def positions_by_rank(variant, d, p):
    """X[i] = coordinates of the point of Hilbert rank i, for the variant."""
    if variant == "skilling":
        N = 2 ** (d * p)
        return hilbert_decode(np.arange(N, dtype=np.int64), p,
                              d).astype(np.float64)
    if variant == "hrc":
        n = 1 << p
        G = np.array(list(itertools.product(*([np.arange(n, dtype=np.int64)] * d))),
                     dtype=np.int64)
        idx = np.array([hilbert_index_hrc(row, d, p) for row in G],
                       dtype=np.int64)
        order = np.argsort(idx, kind="stable")
        return G[order].astype(np.float64)
    raise ValueError(variant)


def wl2(X, d, gap_fraction, exhaustive_upto=32768):
    """Same estimator as exp06: exhaustive for small N, gap sweep above."""
    N = X.shape[0]
    best2 = 0.0
    arg = None
    if N <= exhaustive_upto:
        chunk = 2048
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            diff = X[s:e, None, :] - X[None, :, :]
            d2 = np.sqrt((diff ** 2).sum(axis=2))
            gi = np.abs(np.arange(s, e)[:, None]
                        - np.arange(N)[None, :]).astype(float)
            with np.errstate(divide="ignore", invalid="ignore"):
                r2 = np.where(gi > 0, d2 ** d / gi, 0.0)
            k = int(np.argmax(r2))
            v = float(r2.ravel()[k])
            if v > best2:
                best2 = v
                i, j = divmod(k, N)
                arg = (int(s + i), int(j))
        method, swept = "exhaustive", N - 1
    else:
        swept = int(gap_fraction * N)
        for g in range(1, swept + 1):
            diff = X[g:] - X[:-g]
            d2sq = (diff ** 2).sum(axis=1)
            v = float(d2sq.max()) ** (d / 2.0) / g
            if v > best2:
                best2 = v
                k = int(np.argmax(d2sq))
                arg = (k, k + g)
        method = "gap-sweep g<=%d" % swept
    return {"WL2": best2, "method": method, "gap_swept": swept,
            "argmax_pair": arg,
            "argmax_gap": None if arg is None else abs(arg[1] - arg[0]),
            "argmax_gap_over_N": None if arg is None
            else abs(arg[1] - arg[0]) / N}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--p3", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--gap-fraction", type=float, default=0.08)
    ap.add_argument("--variants", type=str, nargs="+",
                    default=["skilling", "hrc"])
    args = ap.parse_args()

    out = {"experiment": "exp14_wl2_extrapolation", "params": vars(args),
           "hardware": hardware_record(), "published": {
               "%s_d%d" % k: v for k, v in PUBLISHED.items()},
           "series": {}, "extrapolation": {}}

    for d, ps in ((2, args.p2), (3, args.p3)):
        for variant in args.variants:
            key = "%s_d%d" % (variant, d)
            rows = []
            print("=== %s, d = %d ===" % (variant, d))
            for p in ps:
                if d * p > 20:
                    continue
                X = positions_by_rank(variant, d, p)
                r = wl2(X, d, args.gap_fraction)
                r.update({"p": p, "N": int(X.shape[0])})
                rows.append(r)
                print("   p=%d N=%7d  WL2=%10.4f  (%s, argmax gap %s = %.4f N)"
                      % (p, r["N"], r["WL2"], r["method"],
                         r["argmax_gap"], r["argmax_gap_over_N"] or 0.0))
                sys.stdout.flush()
            out["series"][key] = rows
            lim = geometric_limit([r["p"] for r in rows],
                                  [r["WL2"] for r in rows])
            out["extrapolation"][key] = lim
            if lim:
                pub = PUBLISHED.get((variant, d))
                msg = "   extrapolated limit L = %.3f  (r = %.4f, orders %s)" \
                      % (lim["limit"], lim["ratio_r"], lim["orders_used"])
                if pub:
                    err = 100.0 * abs(lim["limit"] - pub["value"]) / pub["value"]
                    msg += "\n   published %.1f  ->  relative error %.2f %%" \
                           % (pub["value"], err)
                    lim["published"] = pub["value"]
                    lim["relative_error_pct"] = err
                print(msg)
            else:
                print("   extrapolation not applicable (non-monotone or too few orders)")
            print()

    # ------------------------------------------------------------------
    print("=== Verdict ===")
    calib = []
    for key, lim in out["extrapolation"].items():
        if lim and "relative_error_pct" in lim:
            calib.append((key, lim["relative_error_pct"]))
            print("   calibration %-14s error %.2f %% against published"
                  % (key, lim["relative_error_pct"]))
    sk3 = out["extrapolation"].get("skilling_d3")
    if sk3:
        print("   Skilling d=3 (no published value): L = %.2f" % sk3["limit"])
    hr3 = out["extrapolation"].get("hrc_d3")
    if sk3 and hr3:
        print("   ratio Skilling/H&RC = %.3f  -- the widely implemented"
              " variant is that much worse" % (sk3["limit"] / hr3["limit"]))
    out["calibration_errors_pct"] = dict(calib)

    save_json(out, "exp14_wl2_extrapolation.json")


if __name__ == "__main__":
    main()
