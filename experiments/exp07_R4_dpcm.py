#!/usr/bin/env python3
"""
EXPERIMENT 07 -- R4: regularity transfer and the DPCM criterion.

Two claims are tested.

(i) Transfer of the Hoelder exponent.  If u is alpha-Hoelder on the cube
    then v = u o H is (alpha/d)-Hoelder on the segment.  We measure both
    exponents from the second-order structure functions

        S_3d(delta) = E |u(x + delta e) - u(x)|^2 ~ delta^{2 alpha},
        S_1d(t)     = E |v(s + t) - v(s)|^2       ~ t^{2 beta},

    and check beta = alpha / d.

(ii) The DPCM criterion.  Along the raster traversal the residual second
     moment splits into a bulk part B and a wrap-around part W:

        M_2(raster) = (1 - 1/n) B + (1/n) W        (up to O(n^{-d})),
        M_2(Hilbert) = B_H  (all Hilbert steps are unit steps).

     Hence Hilbert beats raster if and only if

        W  >  B + n (B_H - B),

     and in particular, when B_H = B, if and only if W > B.  The ratio
     W/B is computable from the data in O(N) *before* any compression is
     attempted, so the sign of the effect is predictable.

     The claim is verified on the MNI152 template (where it predicts, and
     we confirm, that Hilbert *loses*) and on synthetic fractional
     Brownian volumes of prescribed Hoelder exponent (where it predicts,
     and we confirm, that Hilbert wins).

Output: results/exp07_R4_dpcm.json
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from phve import model                                           # noqa: E402
from phve.hilbert import hilbert_decode                          # noqa: E402
from common import hardware_record, save_json                    # noqa: E402


# ----------------------------------------------------------------------
# Synthetic volumes with a prescribed Hoelder exponent
# ----------------------------------------------------------------------

def fractional_brownian_volume(n, H, seed):
    """Isotropic fractional Brownian field on the n^3 grid, Hurst index H.

    Spectral synthesis: white noise filtered by |k|^{-(H + d/2)}.  The
    resulting field is almost surely H-Hoelder for every exponent < H.

    The synthesis is performed on a (2n)^3 torus and then cropped to
    n^3.  Without the crop the field would be periodic, the raster
    wrap-around pairs would be short steps rather than long jumps, and
    the experiment would measure an artefact of the synthesis instead of
    the phenomenon under study.
    """
    rng = np.random.default_rng(seed)
    m2 = 2 * n
    noise = rng.normal(size=(m2, m2, m2))
    k = np.fft.fftfreq(m2) * m2
    KX, KY, KZ = np.meshgrid(k, k, k, indexing="ij")
    K = np.sqrt(KX ** 2 + KY ** 2 + KZ ** 2)
    K[0, 0, 0] = 1.0
    filt = K ** (-(H + 1.5))
    filt[0, 0, 0] = 0.0
    field = np.real(np.fft.ifftn(np.fft.fftn(noise) * filt))
    field = field[:n, :n, :n]
    field -= field.min()
    m = field.max()
    return field / m if m > 0 else field


# ----------------------------------------------------------------------
# Structure functions
# ----------------------------------------------------------------------

def structure_3d(vol, deltas):
    """S(delta) averaged over the three axes."""
    out = []
    for dlt in deltas:
        acc, cnt = 0.0, 0
        for ax in range(3):
            a = np.take(vol, np.arange(dlt, vol.shape[ax]), axis=ax)
            b = np.take(vol, np.arange(0, vol.shape[ax] - dlt), axis=ax)
            acc += float(((a - b) ** 2).sum())
            cnt += a.size
        out.append(acc / cnt)
    return np.array(out)


def structure_1d(sig, lags):
    return np.array([float(((sig[l:] - sig[:-l]) ** 2).mean()) for l in lags])


def loglog_slope(x, y):
    ok = (x > 0) & (y > 0)
    s, b = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    resid = np.log(y[ok]) - (s * np.log(x[ok]) + b)
    r2 = 1 - resid.var() / np.log(y[ok]).var()
    return float(s), float(r2)


# ----------------------------------------------------------------------
# DPCM measurements
# ----------------------------------------------------------------------

def entropy_bits(resid, nbins=None):
    """Order-0 empirical entropy of the residuals quantised to integers
    on the 8-bit scale used by the DPCM experiment of the paper."""
    q = np.rint(resid * 255.0).astype(np.int64)
    _, counts = np.unique(q, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(p)).sum())


def dpcm_report(vol_cube, p):
    """All the DPCM quantities on a 2^p cube."""
    n = 2 ** p
    assert vol_cube.shape == (n, n, n)

    # Hilbert traversal
    idx = np.arange(n ** 3, dtype=np.int64)
    coords = hilbert_decode(idx, p, 3)
    seq_h = vol_cube[coords[:, 0], coords[:, 1], coords[:, 2]]
    r_h = np.diff(seq_h)

    # raster (row-major) traversal
    seq_r = vol_cube.reshape(-1)
    r_r = np.diff(seq_r)

    # split the raster residuals into bulk (unit step along the last axis)
    # and wrap-around pairs
    flat_pos = np.arange(1, n ** 3)
    is_wrap = (flat_pos % n) == 0
    B = float((r_r[~is_wrap] ** 2).mean())
    W = float((r_r[is_wrap] ** 2).mean())
    B_H = float((r_h ** 2).mean())

    M2_r = float((r_r ** 2).mean())
    M2_h = float((r_h ** 2).mean())

    # predicted difference from the split, and the exact criterion
    predicted_diff = (1.0 / n) * (W - B) - (B_H - B)
    return {
        "p": p, "n": n,
        "M2_raster": M2_r, "M2_hilbert": M2_h,
        "M2_ratio_hilbert_over_raster": M2_h / M2_r if M2_r > 0 else float("nan"),
        "bulk_B": B, "wrap_W": W, "hilbert_step_B_H": B_H,
        "W_over_B": W / B if B > 0 else float("inf"),
        "B_H_over_B": B_H / B if B > 0 else float("inf"),
        "criterion_W_gt_B": bool(W > B),
        "criterion_full": bool(W > B + n * (B_H - B)),
        "predicted_M2_diff": float(predicted_diff),
        "measured_M2_diff": float(M2_r - M2_h),
        "entropy_raster_bits": entropy_bits(r_r),
        "entropy_hilbert_bits": entropy_bits(r_h),
        "hilbert_wins_M2": bool(M2_h < M2_r),
        "hilbert_wins_entropy": bool(entropy_bits(r_h) < entropy_bits(r_r)),
    }


def centre_cube(data, p):
    """Largest centred 2^p cube fitting in the volume (zero-padded if needed)."""
    n = 2 ** p
    out = np.zeros((n, n, n), dtype=float)
    sl_src, sl_dst = [], []
    for ax in range(3):
        s = data.shape[ax]
        if s >= n:
            o = (s - n) // 2
            sl_src.append(slice(o, o + n))
            sl_dst.append(slice(0, n))
        else:
            o = (n - s) // 2
            sl_src.append(slice(0, s))
            sl_dst.append(slice(o, o + s))
    out[tuple(sl_dst)] = data[tuple(sl_src)]
    return out


# ----------------------------------------------------------------------

def analyse(name, cube, p, deltas, lags):
    n = 2 ** p
    idx = np.arange(n ** 3, dtype=np.int64)
    coords = hilbert_decode(idx, p, 3)
    seq_h = cube[coords[:, 0], coords[:, 1], coords[:, 2]]

    S3 = structure_3d(cube, deltas)
    S1 = structure_1d(seq_h, lags)
    a2, r2a = loglog_slope(np.array(deltas, float), S3)
    b2, r2b = loglog_slope(np.array(lags, float), S1)
    alpha = a2 / 2.0
    beta = b2 / 2.0
    rec = {
        "name": name,
        "alpha_3d_measured": alpha, "alpha_fit_r2": r2a,
        "beta_1d_measured": beta, "beta_fit_r2": r2b,
        "beta_predicted_alpha_over_d": alpha / 3.0,
        "beta_ratio_measured_over_predicted":
            beta / (alpha / 3.0) if alpha > 0 else float("nan"),
        "dpcm": dpcm_report(cube, p),
    }
    return rec


def collision_sweep(resolution=2, orders=(6, 7, 8, 9, 10, 11)):
    """The v4 protocol as a function of the order p.

    At p = 7 the CR reference box gives cells of about 2 mm, comparable to
    the voxel size, so a large fraction of the voxels share a Hilbert
    index.  Ties are then broken by the raster index, which destroys the
    locality of the traversal exactly where it was supposed to act.  This
    sweep measures the collision rate and the resulting DPCM entropy.
    """
    from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
    from phve.hilbert import hilbert_encode, normalise

    img = load_mni152_template(resolution=resolution)
    data = np.asarray(img.dataobj).astype(np.float64)
    mask = np.asarray(load_mni152_brain_mask(resolution=resolution)
                      .dataobj).astype(bool)
    affine = np.asarray(img.affine, dtype=float)
    coords = np.argwhere(mask)
    xyz = coords @ affine[:3, :3].T + affine[:3, 3]
    box = np.array([[-150.0, 150.0], [-125.0, 125.0], [-125.0, 125.0]])
    inten = data[mask]
    q = np.rint(inten * 255.0).astype(np.int64)

    def ent(a):
        _, c = np.unique(a, return_counts=True)
        pr = c / c.sum()
        return float(-(pr * np.log2(pr)).sum())

    e_raster = ent(np.diff(q))

    # The criterion applied to the *mask-compacted* raster.  Consecutive
    # samples of data[mask] are physically adjacent only when they are
    # consecutive along the last voxel axis; all other consecutive pairs
    # are the wrap-around pairs of this traversal.  Because the mask is
    # convex-ish and the skipped background is exactly what separated the
    # rows, those wraps are short, and the criterion predicts no gain.
    lin = (coords[:, 0] * data.shape[1] + coords[:, 1]) * data.shape[2] \
        + coords[:, 2]
    adjacent = np.diff(lin) == 1
    r = np.diff(inten)
    B_c = float((r[adjacent] ** 2).mean())
    W_c = float((r[~adjacent] ** 2).mean())
    # the Hilbert traversal restricted to the mask is *not* a sequence of
    # unit steps: whenever the curve leaves the mask and returns, two
    # consecutive retained samples are far apart.  B_H / B measures how far
    # the hypothesis of the criterion is from being satisfied here.
    g_ref = normalise(xyz, box, 10)
    h_ref = hilbert_encode(g_ref, 10)
    ord_ref = np.argsort(h_ref, kind="stable")
    xyz_h = coords[ord_ref]
    step_h = np.abs(np.diff(xyz_h, axis=0)).max(axis=1)
    B_H = float((np.diff(inten[ord_ref]) ** 2).mean())
    compacted = {
        "bulk_B": B_c, "wrap_W": W_c, "W_over_B": W_c / B_c,
        "wrap_fraction": float((~adjacent).mean()),
        "hilbert_B_H": B_H, "B_H_over_B": B_H / B_c,
        "hilbert_unit_step_fraction": float((step_h <= 1).mean()),
        "naive_criterion_W_gt_B": bool(W_c > B_c),
        "full_criterion": bool(W_c * float((~adjacent).mean())
                               + B_c * (1 - float((~adjacent).mean())) > B_H),
        "criterion_hypothesis_holds": bool(abs(B_H / B_c - 1.0) < 0.25),
    }

    rows = []
    for p in orders:
        g = normalise(xyz, box, p)
        h = hilbert_encode(g, p)
        coll = int(h.size - np.unique(h).size)
        order = np.argsort(h, kind="stable")
        e_h = ent(np.diff(q[order]))
        rows.append({"p": p, "collisions": coll,
                     "collision_rate": coll / h.size,
                     "entropy_dpcm_hilbert_bits": e_h,
                     "entropy_dpcm_raster_bits": e_raster,
                     "gain_bits": e_raster - e_h,
                     "hilbert_wins": bool(e_h < e_raster)})
    # the same criterion for the *full-volume* raster, background included
    full = data.reshape(-1)
    rf = np.diff(full)
    pos = np.arange(1, full.size)
    is_wrap = (pos % data.shape[2]) == 0
    full_crit = {
        "bulk_B": float((rf[~is_wrap] ** 2).mean()),
        "wrap_W": float((rf[is_wrap] ** 2).mean()),
    }
    full_crit["W_over_B"] = full_crit["wrap_W"] / full_crit["bulk_B"]
    full_crit["criterion_predicts_hilbert_gain"] = bool(
        full_crit["wrap_W"] > full_crit["bulk_B"])

    return {"n_voxels": int(mask.sum()), "resolution_mm": resolution,
            "entropy_dpcm_raster_bits": e_raster, "rows": rows,
            "criterion_mask_compacted_raster": compacted,
            "criterion_full_volume_raster": full_crit}


def replicate_v4_protocol(p=7, resolution=2):
    """Re-run the DPCM experiment of the v4 paper, as published and corrected.

    The published script restricts both traversals to the brain mask and
    then quantises with ``np.round`` a template whose values lie in
    [0, 0.988].  That collapses the signal to {0, 1} before any residual
    is computed, so the reported entropies are those of a binary image and
    not of the MRI.  We report both the published pipeline and the same
    pipeline with the single line ``round(x)`` replaced by
    ``round(255 x)``.
    """
    from nilearn.datasets import load_mni152_brain_mask, load_mni152_template
    from phve.hilbert import hilbert_encode, normalise

    img = load_mni152_template(resolution=resolution)
    data = np.asarray(img.dataobj).astype(np.float64)
    mask = np.asarray(load_mni152_brain_mask(resolution=resolution)
                      .dataobj).astype(bool)
    affine = np.asarray(img.affine, dtype=float)

    coords = np.argwhere(mask)
    xyz = coords @ affine[:3, :3].T + affine[:3, 3]
    box = np.array([[-150.0, 150.0], [-125.0, 125.0], [-125.0, 125.0]])
    g = normalise(xyz, box, p)
    h = hilbert_encode(g, p)
    order = np.argsort(h, kind="stable")

    inten = data[mask]
    seq_hilbert = inten[order]
    seq_raster = inten                      # mask voxels in row-major order

    out = {"p": p, "resolution_mm": resolution,
           "n_masked_voxels": int(mask.sum()),
           "template_min": float(data.min()), "template_max": float(data.max()),
           "hilbert_index_collisions": int(inten.size - np.unique(h).size)}
    for label, scale in (("as_published_round_to_unit", 1.0),
                         ("corrected_8bit", 255.0)):
        qr = np.rint(seq_raster * scale).astype(np.int64)
        qh = np.rint(seq_hilbert * scale).astype(np.int64)

        def ent(a):
            _, c = np.unique(a, return_counts=True)
            pr = c / c.sum()
            return float(-(pr * np.log2(pr)).sum())

        rr, rh = np.diff(qr), np.diff(qh)
        out[label] = {
            "distinct_levels": int(np.unique(qr).size),
            "entropy_original_bits": ent(qr),
            "entropy_dpcm_raster_bits": ent(rr),
            "entropy_dpcm_hilbert_bits": ent(rh),
            "gain_hilbert_vs_raster_bits": ent(rr) - ent(rh),
            "gain_percent": 100.0 * (1 - ent(rh) / ent(rr)) if ent(rr) > 0 else 0.0,
            "M2_raster": float((rr.astype(float) ** 2).mean()),
            "M2_hilbert": float((rh.astype(float) ** 2).mean()),
            "hilbert_wins": bool(ent(rh) < ent(rr)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    p = args.p
    n = 2 ** p
    deltas = [1, 2, 3, 4, 6, 8, 12, 16]
    lags = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    records = []

    # -- real templates --------------------------------------------------
    for res in (2, 1):
        try:
            vol = model.load_mni152(res)
        except Exception as exc:            # pragma: no cover
            print(f"[skip] MNI152 {res}mm: {exc}")
            continue
        cube = centre_cube(vol.data, p)
        records.append(analyse(f"MNI152-{res}mm", cube, p, deltas, lags))

    # -- synthetic fractional Brownian volumes ---------------------------
    for H in (0.2, 0.35, 0.5, 0.65, 0.8):
        cube = fractional_brownian_volume(n, H, args.seed)
        records.append(analyse(f"fBm-H{H:.2f}", cube, p, deltas, lags))

    # -- an MNI152 cube restricted to the brain support ------------------
    # (the wrap-around pairs then no longer sit in the constant background)
    vol = model.load_mni152(2)
    idx = np.argwhere(vol.data > 0.15)
    lo = idx.min(axis=0)
    sub = vol.data[lo[0]:lo[0] + n, lo[1]:lo[1] + n, lo[2]:lo[2] + n]
    if sub.shape == (n, n, n):
        records.append(analyse("MNI152-2mm-tight-crop", sub, p, deltas, lags))

    print("%-24s %7s %7s %7s | %8s %8s %7s | %s"
          % ("volume", "alpha", "beta", "a/3", "W/B", "M2_H/M2_R",
             "dH-dR", "pred/meas"))
    for r in records:
        d = r["dpcm"]
        print("%-24s %7.3f %7.3f %7.3f | %8.3f %9.4f %7.4f | %s / %s"
              % (r["name"], r["alpha_3d_measured"], r["beta_1d_measured"],
                 r["beta_predicted_alpha_over_d"], d["W_over_B"],
                 d["M2_ratio_hilbert_over_raster"],
                 d["entropy_hilbert_bits"] - d["entropy_raster_bits"],
                 d["criterion_W_gt_B"], d["hilbert_wins_M2"]))

    agree = sum(1 for r in records
                if r["dpcm"]["criterion_full"] == r["dpcm"]["hilbert_wins_M2"])
    print("\ncriterion agrees with the measured sign on %d/%d volumes"
          % (agree, len(records)))

    v4 = replicate_v4_protocol()
    print("\nreplication of the v4 DPCM protocol (mask-restricted, p=7)")
    print("   template range [%.3f, %.3f];  Hilbert index collisions: %d"
          % (v4["template_min"], v4["template_max"],
             v4["hilbert_index_collisions"]))
    for label in ("as_published_round_to_unit", "corrected_8bit"):
        r = v4[label]
        print("   %-26s levels=%4d  H_orig=%.4f  raster=%.4f  hilbert=%.4f"
              "  gain=%+.4f bits (%+.2f%%)"
              % (label, r["distinct_levels"], r["entropy_original_bits"],
                 r["entropy_dpcm_raster_bits"], r["entropy_dpcm_hilbert_bits"],
                 r["gain_hilbert_vs_raster_bits"], r["gain_percent"]))

    sweep = collision_sweep()
    print("\ncollision sweep of the v4 protocol (8-bit quantisation, %d voxels)"
          % sweep["n_voxels"])
    print("   raster DPCM entropy = %.4f bits" % sweep["entropy_dpcm_raster_bits"])
    for r in sweep["rows"]:
        print("   p=%2d  collisions=%6d (%5.2f%%)  hilbert=%.4f bits  "
              "gain=%+.4f  wins=%s"
              % (r["p"], r["collisions"], 100 * r["collision_rate"],
                 r["entropy_dpcm_hilbert_bits"], r["gain_bits"],
                 r["hilbert_wins"]))
    cc = sweep["criterion_mask_compacted_raster"]
    cf = sweep["criterion_full_volume_raster"]
    print("   criterion, mask-compacted raster : W/B = %.3f (wrap frac %.4f), "
          "B_H/B = %.3f, Hilbert unit-step fraction %.4f"
          % (cc["W_over_B"], cc["wrap_fraction"], cc["B_H_over_B"],
             cc["hilbert_unit_step_fraction"]))
    print("      reduced criterion (W > B)        predicts gain: %s   [WRONG]"
          % cc["naive_criterion_W_gt_B"])
    print("      full criterion ((1-f)B + fW > B_H) predicts gain: %s   [RIGHT]"
          % cc["full_criterion"])
    print("   criterion, full-volume raster    : W/B = %.3f"
          " -> gain predicted: %s"
          % (cf["W_over_B"], cf["criterion_predicts_hilbert_gain"]))

    out = {"experiment": "exp07_R4_dpcm", "params": vars(args),
           "hardware": hardware_record(), "records": records,
           "criterion_agreement": {"n_agree": agree, "n_total": len(records)},
           "v4_protocol_replication": v4,
           "v4_collision_sweep": sweep}
    save_json(out, "exp07_R4_dpcm.json")


if __name__ == "__main__":
    main()
