#!/usr/bin/env python3
"""
EXPERIMENT 05 -- R1: anisotropy correction and the per-axis order rule.

Two things are measured.

(1) The kappa-correction.  On boxes of increasing anisotropy
    kappa = max_i L_i / min_i L_i we build an unstructured mesh inside an
    ellipsoid inscribed in the box and measure the index-gap functionals
    under the cubic-order PHVE numbering.  The claim under test is that
    the grid distance of a physical edge of length h is
        r = h / Delta_min = kappa * h / Delta_max,
    so that the index-gap bound carries a factor kappa^d that the cubic
    statement of the published theorem omits.

(2) The selection rule.  With the total budget sum_i p_i held fixed, the
    per-axis orders p_i are chosen to equalise the cell edge lengths
    Delta_i = L_i 2^{-p_i}.  The predicted gain over the uniform choice is
        rho^d,  rho = (prod_j L_j)^{1/d} / min_i L_i,
    for the max-type bound and rho for the mean-type bound.  Both are
    compared with the measurement.

The rule is finally applied to the full-body reference box
2000 x 600 x 400 mm of the anatomical Reference Family.

Output: results/exp05_R1_anisotropy.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import mesh as meshmod                                 # noqa: E402
from phve.hilbert import choose_orders, phve_order, phve_order_aniso  # noqa: E402
from common import hardware_record, save_json                    # noqa: E402


def gap_stats(edges, perm, n):
    inv = np.empty(n, dtype=np.int64)
    inv[perm] = np.arange(n)
    d = np.abs(inv[edges[:, 0]] - inv[edges[:, 1]])
    return {"max": int(d.max()), "mean": float(d.mean()),
            "median": float(np.median(d)), "p99": float(np.percentile(d, 99)),
            "p999": float(np.percentile(d, 99.9))}


def ellipsoid_mask(box):
    c = box.mean(axis=1)
    r = (box[:, 1] - box[:, 0]) / 2.0

    def mask(pts):
        q = (pts - c) / r
        return np.einsum("ij,ij->i", q, q) <= 1.0
    return mask


def run_case(L, n_target, total_bits, seed):
    box = np.stack([np.zeros(3), np.asarray(L, dtype=float)], axis=1)
    mesh = meshmod.build_mesh(ellipsoid_mask(box), box, n_target, seed=seed)
    n = mesh.n_vertices
    edges = mesh.edges()
    h = np.linalg.norm(mesh.points[edges[:, 0]] - mesh.points[edges[:, 1]],
                       axis=1)

    Lv = np.asarray(L, dtype=float)
    kappa = float(Lv.max() / Lv.min())
    G = float(np.exp(np.mean(np.log(Lv))))
    rho = G / Lv.min()

    p_uni = total_bits // 3
    orders = choose_orders(box, total_bits)

    perm_u = phve_order(mesh.points, box, p_uni)
    perm_a = phve_order_aniso(mesh.points, box, orders)

    Delta_uni = Lv / 2 ** p_uni
    Delta_ani = Lv / 2 ** orders

    return {
        "L": Lv.tolist(), "kappa": kappa, "G": G, "rho": rho,
        "N": int(n), "n_edges": int(edges.shape[0]),
        "h_median": float(np.median(h)), "h_max": float(h.max()),
        "total_bits": int(total_bits),
        "p_uniform": int(p_uni), "p_aniso": orders.tolist(),
        "Delta_uniform": Delta_uni.tolist(),
        "Delta_aniso": Delta_ani.tolist(),
        "aniso_spread_uniform": float(Delta_uni.max() / Delta_uni.min()),
        "aniso_spread_aniso": float(Delta_ani.max() / Delta_ani.min()),
        "r_pred_uniform": float(np.median(h) / Delta_uni.min()),
        "r_pred_aniso": float(np.median(h) / Delta_ani.min()),
        # continuum (non-integer) prediction
        "predicted_gain_max_rho_d": float(rho ** 3),
        "predicted_gain_mean_rho": float(rho),
        # prediction actually achievable with integer per-axis orders
        "achieved_ratio_Delta_min": float(Delta_ani.min() / Delta_uni.min()),
        "achieved_gain_max_bound": float((Delta_ani.min() / Delta_uni.min()) ** 3),
        "achieved_gain_mean_bound": float(Delta_ani.min() / Delta_uni.min()),
        "gap_uniform": gap_stats(edges, perm_u, n),
        "gap_aniso": gap_stats(edges, perm_a, n),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-target", type=int, default=20000)
    ap.add_argument("--total-bits", type=int, default=30)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cases = []
    for kappa in (1, 2, 3, 5, 8, 12, 20):
        # L = (kappa, sqrt(kappa), 1) normalised so that max_i L_i = 1
        L = np.array([float(kappa), float(np.sqrt(kappa)), 1.0])
        L = L / L.max()
        cases.append(run_case(L, args.n_target, args.total_bits, args.seed))
        c = cases[-1]
        print("kappa=%5.1f rho=%5.2f p_aniso=%-11s | mean gap: uniform %7.1f "
              "-> aniso %7.1f (x%5.2f)  | achievable bound x%5.2f  "
              "| max gap: %6d -> %6d (x%4.2f)"
              % (c["kappa"], c["rho"], str(c["p_aniso"]),
                 c["gap_uniform"]["mean"], c["gap_aniso"]["mean"],
                 c["gap_uniform"]["mean"] / max(c["gap_aniso"]["mean"], 1e-9),
                 c["achieved_gain_mean_bound"],
                 c["gap_uniform"]["max"], c["gap_aniso"]["max"],
                 c["gap_uniform"]["max"] / max(c["gap_aniso"]["max"], 1)))

    # ---- the full-body anatomical box -----------------------------------
    fb = np.array([2000.0, 600.0, 400.0])
    box_fb = np.stack([np.zeros(3), fb], axis=1)
    fb_rows = []
    for total_bits in (24, 30, 36):
        orders = choose_orders(box_fb, total_bits)
        p_uni = total_bits // 3
        D_uni = fb / 2 ** p_uni
        D_ani = fb / 2 ** orders
        G = float(np.exp(np.mean(np.log(fb))))
        rho = G / fb.min()
        achieved = float(D_ani.min() / D_uni.min())
        fb_rows.append({
            "total_bits": total_bits,
            "p_uniform": p_uni, "p_aniso": orders.tolist(),
            "Delta_uniform_mm": D_uni.tolist(),
            "Delta_aniso_mm": D_ani.tolist(),
            "cells_uniform": float(2.0 ** (3 * p_uni)),
            "cells_aniso": float(2.0 ** int(orders.sum())),
            "rho_continuum": rho,
            "gain_max_bound_continuum": rho ** 3,
            "gain_mean_bound_continuum": rho,
            "achieved_ratio_Delta_min": achieved,
            "gain_max_bound_achieved": achieved ** 3,
            "gain_mean_bound_achieved": achieved,
            "kappa": float(fb.max() / fb.min()),
        })
        print("\nFB box 2000x600x400, budget %d bits:" % total_bits)
        print("   uniform p=(%d,%d,%d) -> Delta=(%.3f, %.3f, %.3f) mm  "
              "(Delta_min=%.3f)"
              % (p_uni, p_uni, p_uni, *D_uni, D_uni.min()))
        print("   rule    p=(%d,%d,%d) -> Delta=(%.3f, %.3f, %.3f) mm  "
              "(Delta_min=%.3f)"
              % (*orders, *D_ani, D_ani.min()))
        print("   continuum rho = %.4f (bound / rho^3 = %.2f);  "
              "achieved with integer orders: %.4f (bound / %.2f)"
              % (rho, rho ** 3, achieved, achieved ** 3))

    out = {"experiment": "exp05_R1_anisotropy", "params": vars(args),
           "hardware": hardware_record(), "cases": cases,
           "full_body_box": fb_rows}
    save_json(out, "exp05_R1_anisotropy.json")


if __name__ == "__main__":
    main()
